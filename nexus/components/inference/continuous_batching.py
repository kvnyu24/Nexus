"""
Continuous Batching for efficient LLM inference.

Enables dynamic batching of requests, adding new sequences as others complete
to maximize GPU utilization.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import heapq
import time
from collections import deque

from nexus.core.base import NexusModule


class RequestStatus(Enum):
    """Status of a generation request."""
    PENDING = "pending"          # Waiting to be scheduled
    RUNNING = "running"          # Currently generating
    COMPLETED = "completed"      # Generation finished
    CANCELLED = "cancelled"      # Request cancelled
    FAILED = "failed"            # Request failed


@dataclass
class GenerationRequest:
    """
    Represents a single generation request.

    Attributes:
        request_id: Unique identifier for the request
        input_ids: Input token IDs
        max_new_tokens: Maximum tokens to generate
        stop_token_ids: Token IDs that signal generation stop
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
        top_k: Top-k sampling parameter
        priority: Request priority (higher = more important)
        arrival_time: Timestamp when request was added
    """
    request_id: int
    input_ids: List[int]
    max_new_tokens: int = 256
    stop_token_ids: Optional[List[int]] = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    priority: int = 0
    arrival_time: float = field(default_factory=time.time)

    # Runtime state (managed by batcher)
    status: RequestStatus = RequestStatus.PENDING
    generated_ids: List[int] = field(default_factory=list)
    current_len: int = 0
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    batch_idx: Optional[int] = None


@dataclass
class BatchState:
    """
    State of the current batch being processed.

    Attributes:
        request_ids: List of request IDs in batch
        input_ids: Batched input tensor
        position_ids: Position IDs for each sequence
        attention_mask: Attention mask tensor
        seq_lens: Current sequence lengths
    """
    request_ids: List[int]
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    seq_lens: List[int]


class ContinuousBatcher:
    """
    Continuous batching scheduler for efficient inference.

    Dynamically adds/removes sequences from batch as they complete,
    maximizing GPU utilization. Supports priority-based scheduling
    and preemption.

    Features:
    - Dynamic batch composition
    - Priority-based scheduling
    - Automatic padding/unpadding
    - KV cache management integration
    - Request preemption for high-priority requests
    - Statistics tracking

    Args:
        max_batch_size: Maximum sequences in batch
        max_seq_len: Maximum total sequence length
        kv_cache: KV cache manager to use (optional)
        pad_token_id: Token ID for padding
        eos_token_id: End of sequence token ID
        scheduling_policy: 'fcfs' (first-come-first-served) or 'priority'
        enable_preemption: Whether to allow preempting low-priority requests
        device: Device for tensors

    Example:
        >>> batcher = ContinuousBatcher(
        ...     max_batch_size=32,
        ...     max_seq_len=2048,
        ...     pad_token_id=0,
        ...     eos_token_id=2
        ... )
        >>> # Add requests
        >>> batcher.add_request(input_ids=[1, 2, 3], max_new_tokens=100)
        >>> batcher.add_request(input_ids=[1, 4, 5, 6], max_new_tokens=50)
        >>>
        >>> # Process generation steps
        >>> while batcher.has_active_requests():
        ...     batch = batcher.prepare_batch()
        ...     logits = model(batch.input_ids, batch.attention_mask)
        ...     next_tokens = sample(logits)
        ...     completed = batcher.step(next_tokens)
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        kv_cache: Optional[Any] = None,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        scheduling_policy: str = 'fcfs',
        enable_preemption: bool = False,
        device: torch.device = torch.device('cuda')
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.kv_cache = kv_cache
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.scheduling_policy = scheduling_policy
        self.enable_preemption = enable_preemption
        self.device = device

        # Request management
        self._next_request_id = 0
        self._pending_requests: deque = deque()  # FCFS queue
        self._priority_queue: List[Tuple[int, int, GenerationRequest]] = []  # Priority heap
        self._active_requests: Dict[int, GenerationRequest] = {}
        self._completed_requests: Dict[int, GenerationRequest] = {}

        # Batch state
        self._current_batch: Optional[BatchState] = None
        self._batch_request_map: Dict[int, int] = {}  # request_id -> batch_idx

        # Statistics
        self._total_requests = 0
        self._total_tokens_generated = 0
        self._total_time = 0.0

    def add_request(
        self,
        input_ids: List[int],
        max_new_tokens: int = 256,
        stop_token_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        priority: int = 0
    ) -> int:
        """
        Add new request to batch.

        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum tokens to generate
            stop_token_ids: Token IDs that signal generation stop
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            priority: Request priority (higher = more important)

        Returns:
            Request ID for tracking
        """
        request_id = self._next_request_id
        self._next_request_id += 1

        request = GenerationRequest(
            request_id=request_id,
            input_ids=list(input_ids),
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids or [self.eos_token_id],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            priority=priority,
            current_len=len(input_ids)
        )

        if self.scheduling_policy == 'priority':
            # Use negative priority for min-heap (higher priority = lower value)
            heapq.heappush(
                self._priority_queue,
                (-priority, request.arrival_time, request)
            )
        else:
            self._pending_requests.append(request)

        self._total_requests += 1
        return request_id

    def _get_next_pending(self) -> Optional[GenerationRequest]:
        """Get next pending request based on scheduling policy."""
        if self.scheduling_policy == 'priority':
            if self._priority_queue:
                _, _, request = heapq.heappop(self._priority_queue)
                return request
        else:
            if self._pending_requests:
                return self._pending_requests.popleft()
        return None

    def _can_add_to_batch(self, request: GenerationRequest) -> bool:
        """Check if request can be added to current batch."""
        if len(self._active_requests) >= self.max_batch_size:
            return False

        # Check if sequence would exceed max length
        total_len = request.current_len + request.max_new_tokens
        if total_len > self.max_seq_len:
            return False

        return True

    def schedule(self) -> int:
        """
        Schedule pending requests into active batch.

        Returns:
            Number of newly scheduled requests
        """
        scheduled = 0

        while True:
            # Check if we can add more requests
            if len(self._active_requests) >= self.max_batch_size:
                break

            # Get next pending request
            request = self._get_next_pending()
            if request is None:
                break

            if not self._can_add_to_batch(request):
                # Put back in queue if can't schedule
                if self.scheduling_policy == 'priority':
                    heapq.heappush(
                        self._priority_queue,
                        (-request.priority, request.arrival_time, request)
                    )
                else:
                    self._pending_requests.appendleft(request)
                break

            # Activate request
            request.status = RequestStatus.RUNNING
            request.start_time = time.time()
            self._active_requests[request.request_id] = request
            scheduled += 1

        return scheduled

    def prepare_batch(self) -> Optional[BatchState]:
        """
        Prepare batch tensors for model forward pass.

        Returns:
            BatchState with batched tensors, or None if no active requests
        """
        if not self._active_requests:
            self._current_batch = None
            return None

        requests = list(self._active_requests.values())
        batch_size = len(requests)

        # Find max sequence length in batch
        max_len = max(r.current_len for r in requests)

        # Prepare batch tensors
        input_ids = torch.full(
            (batch_size, 1),  # Only need last token for generation
            self.pad_token_id,
            dtype=torch.long,
            device=self.device
        )

        position_ids = torch.zeros(
            (batch_size, 1),
            dtype=torch.long,
            device=self.device
        )

        # Attention mask for full sequence (including cached KV)
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.bool,
            device=self.device
        )

        seq_lens = []
        request_ids = []

        for batch_idx, request in enumerate(requests):
            # Get last token (or full sequence for first step)
            if request.generated_ids:
                input_ids[batch_idx, 0] = request.generated_ids[-1]
            else:
                input_ids[batch_idx, 0] = request.input_ids[-1]

            # Position is current sequence length - 1
            position_ids[batch_idx, 0] = request.current_len - 1

            # Mask valid positions
            attention_mask[batch_idx, :request.current_len] = True

            seq_lens.append(request.current_len)
            request_ids.append(request.request_id)
            self._batch_request_map[request.request_id] = batch_idx

        self._current_batch = BatchState(
            request_ids=request_ids,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            seq_lens=seq_lens
        )

        return self._current_batch

    def prepare_prefill_batch(self) -> Optional[BatchState]:
        """
        Prepare batch for prefill phase (processing full input).

        Returns:
            BatchState with full input sequences for prefill
        """
        # Get requests that haven't started generation yet
        prefill_requests = [
            r for r in self._active_requests.values()
            if not r.generated_ids
        ]

        if not prefill_requests:
            return None

        batch_size = len(prefill_requests)
        max_len = max(len(r.input_ids) for r in prefill_requests)

        # Prepare padded batch
        input_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device
        )

        position_ids = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=self.device
        )

        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.bool,
            device=self.device
        )

        seq_lens = []
        request_ids = []

        for batch_idx, request in enumerate(prefill_requests):
            seq_len = len(request.input_ids)
            input_ids[batch_idx, :seq_len] = torch.tensor(
                request.input_ids, dtype=torch.long, device=self.device
            )
            position_ids[batch_idx, :seq_len] = torch.arange(
                seq_len, dtype=torch.long, device=self.device
            )
            attention_mask[batch_idx, :seq_len] = True

            seq_lens.append(seq_len)
            request_ids.append(request.request_id)

        return BatchState(
            request_ids=request_ids,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            seq_lens=seq_lens
        )

    def step(
        self,
        next_tokens: torch.Tensor
    ) -> List[GenerationRequest]:
        """
        Process one generation step for all active sequences.

        Args:
            next_tokens: Next tokens for each sequence in batch (batch_size,)

        Returns:
            List of completed requests
        """
        if self._current_batch is None:
            return []

        completed = []
        next_tokens = next_tokens.cpu().tolist()

        for batch_idx, request_id in enumerate(self._current_batch.request_ids):
            if request_id not in self._active_requests:
                continue

            request = self._active_requests[request_id]
            next_token = next_tokens[batch_idx]

            # Add generated token
            request.generated_ids.append(next_token)
            request.current_len += 1
            self._total_tokens_generated += 1

            # Check stopping conditions
            should_stop = False

            # Check for stop tokens
            if next_token in request.stop_token_ids:
                should_stop = True

            # Check max length
            if len(request.generated_ids) >= request.max_new_tokens:
                should_stop = True

            # Check total sequence length
            if request.current_len >= self.max_seq_len:
                should_stop = True

            if should_stop:
                request.status = RequestStatus.COMPLETED
                request.finish_time = time.time()
                completed.append(request)

                # Move to completed
                del self._active_requests[request_id]
                self._completed_requests[request_id] = request

                if request_id in self._batch_request_map:
                    del self._batch_request_map[request_id]

        # Schedule new requests to fill empty slots
        self.schedule()

        return completed

    def has_active_requests(self) -> bool:
        """Check if there are active or pending requests."""
        has_pending = (
            len(self._pending_requests) > 0 or
            len(self._priority_queue) > 0
        )
        return len(self._active_requests) > 0 or has_pending

    def get_request_status(self, request_id: int) -> Optional[RequestStatus]:
        """Get status of a request."""
        if request_id in self._active_requests:
            return self._active_requests[request_id].status
        if request_id in self._completed_requests:
            return self._completed_requests[request_id].status
        return None

    def get_request(self, request_id: int) -> Optional[GenerationRequest]:
        """Get request by ID."""
        if request_id in self._active_requests:
            return self._active_requests[request_id]
        if request_id in self._completed_requests:
            return self._completed_requests[request_id]
        return None

    def cancel_request(self, request_id: int) -> bool:
        """
        Cancel a pending or active request.

        Args:
            request_id: ID of request to cancel

        Returns:
            True if request was cancelled, False if not found
        """
        # Check active requests
        if request_id in self._active_requests:
            request = self._active_requests.pop(request_id)
            request.status = RequestStatus.CANCELLED
            request.finish_time = time.time()
            self._completed_requests[request_id] = request

            if request_id in self._batch_request_map:
                del self._batch_request_map[request_id]
            return True

        # Check pending queue
        for i, req in enumerate(self._pending_requests):
            if req.request_id == request_id:
                req.status = RequestStatus.CANCELLED
                self._pending_requests.remove(req)
                self._completed_requests[request_id] = req
                return True

        return False

    def preempt_request(self, request_id: int) -> bool:
        """
        Preempt an active request (move back to pending).

        Used to free resources for higher-priority requests.

        Args:
            request_id: ID of request to preempt

        Returns:
            True if request was preempted, False if not found
        """
        if not self.enable_preemption:
            return False

        if request_id not in self._active_requests:
            return False

        request = self._active_requests.pop(request_id)
        request.status = RequestStatus.PENDING

        # Re-add to pending queue
        if self.scheduling_policy == 'priority':
            heapq.heappush(
                self._priority_queue,
                (-request.priority, request.arrival_time, request)
            )
        else:
            self._pending_requests.appendleft(request)

        if request_id in self._batch_request_map:
            del self._batch_request_map[request_id]

        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get batch statistics.

        Returns:
            Dictionary with batching statistics
        """
        pending_count = (
            len(self._pending_requests) + len(self._priority_queue)
        )

        return {
            'total_requests': self._total_requests,
            'pending_requests': pending_count,
            'active_requests': len(self._active_requests),
            'completed_requests': len(self._completed_requests),
            'total_tokens_generated': self._total_tokens_generated,
            'avg_batch_size': (
                len(self._active_requests) if self._active_requests else 0
            ),
            'max_batch_size': self.max_batch_size,
            'scheduling_policy': self.scheduling_policy
        }

    def clear_completed(self):
        """Clear completed requests to free memory."""
        self._completed_requests.clear()

    def reset(self):
        """Reset batcher state."""
        self._pending_requests.clear()
        self._priority_queue.clear()
        self._active_requests.clear()
        self._completed_requests.clear()
        self._current_batch = None
        self._batch_request_map.clear()


class IterationLevelBatcher(ContinuousBatcher):
    """
    Iteration-level batching with chunked prefill support.

    Extends ContinuousBatcher with support for:
    - Chunked prefill to limit prefill batch size
    - Iteration-level scheduling for better latency
    - Memory budget management

    Used by: Sarathi, Splitwise

    Args:
        max_batch_size: Maximum sequences in batch
        max_seq_len: Maximum total sequence length
        max_prefill_tokens: Maximum tokens to prefill per iteration
        max_decode_tokens: Maximum tokens to decode per iteration
        **kwargs: Additional arguments passed to ContinuousBatcher
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        max_prefill_tokens: int = 2048,
        max_decode_tokens: int = 512,
        **kwargs
    ):
        super().__init__(max_batch_size, max_seq_len, **kwargs)

        self.max_prefill_tokens = max_prefill_tokens
        self.max_decode_tokens = max_decode_tokens

        # Track prefill progress for chunked prefill
        self._prefill_progress: Dict[int, int] = {}  # request_id -> prefilled_tokens

    def prepare_iteration_batch(
        self
    ) -> Tuple[Optional[BatchState], Optional[BatchState]]:
        """
        Prepare batches for one iteration with budget constraints.

        Returns:
            Tuple of (prefill_batch, decode_batch)
        """
        prefill_batch = self._prepare_chunked_prefill()
        decode_batch = self._prepare_decode_batch()

        return prefill_batch, decode_batch

    def _prepare_chunked_prefill(self) -> Optional[BatchState]:
        """Prepare chunked prefill batch within token budget."""
        # Get requests needing prefill
        prefill_requests = []
        total_tokens = 0

        for request in self._active_requests.values():
            if not request.generated_ids:
                prefilled = self._prefill_progress.get(request.request_id, 0)
                remaining = len(request.input_ids) - prefilled

                if remaining > 0:
                    # Calculate chunk size
                    chunk_size = min(
                        remaining,
                        self.max_prefill_tokens - total_tokens
                    )

                    if chunk_size > 0:
                        prefill_requests.append((request, prefilled, chunk_size))
                        total_tokens += chunk_size

                    if total_tokens >= self.max_prefill_tokens:
                        break

        if not prefill_requests:
            return None

        # Build batch
        batch_size = len(prefill_requests)
        max_chunk = max(chunk for _, _, chunk in prefill_requests)

        input_ids = torch.full(
            (batch_size, max_chunk),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device
        )

        position_ids = torch.zeros(
            (batch_size, max_chunk),
            dtype=torch.long,
            device=self.device
        )

        attention_mask = torch.zeros(
            (batch_size, max_chunk),
            dtype=torch.bool,
            device=self.device
        )

        seq_lens = []
        request_ids = []

        for batch_idx, (request, start_pos, chunk_size) in enumerate(prefill_requests):
            # Get chunk of input
            chunk = request.input_ids[start_pos:start_pos + chunk_size]
            input_ids[batch_idx, :len(chunk)] = torch.tensor(
                chunk, dtype=torch.long, device=self.device
            )

            # Position IDs continue from previous chunk
            position_ids[batch_idx, :len(chunk)] = torch.arange(
                start_pos, start_pos + len(chunk),
                dtype=torch.long, device=self.device
            )

            attention_mask[batch_idx, :len(chunk)] = True

            seq_lens.append(chunk_size)
            request_ids.append(request.request_id)

            # Update prefill progress
            self._prefill_progress[request.request_id] = start_pos + chunk_size

        return BatchState(
            request_ids=request_ids,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            seq_lens=seq_lens
        )

    def _prepare_decode_batch(self) -> Optional[BatchState]:
        """Prepare decode batch within token budget."""
        # Get requests in decode phase
        decode_requests = []
        total_tokens = 0

        for request in self._active_requests.values():
            if request.generated_ids:
                if total_tokens < self.max_decode_tokens:
                    decode_requests.append(request)
                    total_tokens += 1

        if not decode_requests:
            return None

        # Use parent's prepare_batch logic for decode
        # (temporarily filter active requests)
        original_active = self._active_requests.copy()
        self._active_requests = {
            r.request_id: r for r in decode_requests
        }

        batch = self.prepare_batch()

        self._active_requests = original_active
        return batch

# GaussianEditor: Interactive 3D Scene Editing with Gaussian Splatting

## Overview & Motivation

GaussianEditor enables **semantic-aware editing** of 3D scenes represented as Gaussian Splatting. Users can select, move, remove, or modify parts of a 3D scene interactively while maintaining consistency across all views.

### Key Capabilities

1. **Semantic Segmentation**: Automatically identify objects in 3D
2. **Interactive Selection**: Click-to-select objects
3. **3D Transformations**: Move, rotate, scale selected parts
4. **Inpainting**: Remove objects and fill holes
5. **Cloning**: Duplicate objects
6. **Real-Time Preview**: See edits immediately (60+ FPS)

## Theoretical Background

### 3D Semantic Segmentation

Lift 2D segmentation masks to 3D Gaussians:

```
For each training view:
  Segment image → 2D masks
  For each Gaussian:
    Render contribution to each mask
    Assign semantic label via majority vote
```

### Gaussian Hierarchical Grouping

```
Gaussians → Supervoxels → Objects

1. Group nearby Gaussians into supervoxels
2. Merge supervoxels with same semantic label
3. Form object-level entities
```

### Inpainting via Depth-Guided Diffusion

Remove object and fill hole:

```
1. Render without selected Gaussians
2. Detect disoccluded regions (holes)
3. Use depth map + diffusion to inpaint
4. Re-optimize Gaussians in filled regions
```

## Mathematical Formulation

### Semantic Label Assignment

```
For Gaussian i:
  L_i = argmax_label Σ_v w_v^i · M_v^label

where:
w_v^i: Rendering weight of Gaussian i in view v
M_v^label: Binary mask for label in view v
```

### Edit Transformation

For translation, rotation, scale:

```
T(μ_i) = R · s · μ_i + t

where:
μ_i: Original position
R: Rotation matrix
s: Scale factor
t: Translation vector

Updated Gaussian:
μ_i' = T(μ_i)
Σ_i' = R · s · Σ_i · s · R^T  (covariance transforms too)
```

### Inpainting Loss

```
L_inpaint = L_rgb(inpainted) + L_depth(geometry) + L_consistency(multiview)
```

## Implementation Details

### Semantic Segmentation Pipeline

```python
def segment_gaussians(
    gaussians,
    train_views,
    segmentation_model  # e.g., SAM, GroundingDINO
):
    """Assign semantic labels to each Gaussian."""

    # Step 1: Segment all training views
    view_masks = []
    for view in train_views:
        image = render_gaussians(gaussians, view)
        masks = segmentation_model(image)  # [K, H, W]
        view_masks.append(masks)

    # Step 2: Compute contribution of each Gaussian to each mask
    n_gaussians = len(gaussians)
    n_labels = view_masks[0].shape[0]

    label_scores = torch.zeros(n_gaussians, n_labels)

    for view_idx, (view, masks) in enumerate(zip(train_views, view_masks)):
        # Render Gaussian contributions
        contrib = render_gaussian_contributions(gaussians, view)  # [N, H, W]

        for label_idx in range(n_labels):
            mask = masks[label_idx]  # [H, W]

            # Sum contribution within mask
            for gaussian_idx in range(n_gaussians):
                label_scores[gaussian_idx, label_idx] += (
                    contrib[gaussian_idx] * mask
                ).sum()

    # Step 3: Assign labels
    gaussian_labels = torch.argmax(label_scores, dim=1)

    return gaussian_labels
```

### Interactive Selection

```python
class GaussianEditor:
    def __init__(self, gaussians):
        self.gaussians = gaussians
        self.selected_mask = torch.zeros(len(gaussians), dtype=torch.bool)

    def select_by_click(self, click_position, camera):
        """Select object at clicked pixel."""
        # Render Gaussian IDs at click location
        id_map = render_gaussian_ids(self.gaussians, camera)

        x, y = int(click_position[0]), int(click_position[1])
        clicked_id = id_map[y, x]

        if clicked_id >= 0:
            # Select all Gaussians with same semantic label
            clicked_label = self.gaussians.labels[clicked_id]
            self.selected_mask = (self.gaussians.labels == clicked_label)

            print(f"Selected {self.selected_mask.sum()} Gaussians")

    def translate(self, translation_vector):
        """Move selected Gaussians."""
        positions = self.gaussians.get_xyz()
        positions[self.selected_mask] += translation_vector

        self.gaussians._xyz.data = positions

    def rotate(self, rotation_quaternion, center=None):
        """Rotate selected Gaussians around center."""
        if center is None:
            # Rotate around centroid of selection
            center = self.gaussians.get_xyz()[self.selected_mask].mean(dim=0)

        # Update positions
        positions = self.gaussians.get_xyz()
        selected_positions = positions[self.selected_mask] - center

        R = quaternion_to_rotation_matrix(rotation_quaternion)
        rotated_positions = (R @ selected_positions.T).T + center

        positions[self.selected_mask] = rotated_positions

        # Update rotations
        rotations = self.gaussians.get_rotation()
        selected_rotations = rotations[self.selected_mask]

        new_rotations = quaternion_multiply(rotation_quaternion, selected_rotations)
        rotations[self.selected_mask] = new_rotations

        # Update covariances (they depend on rotation + scale)
        self.gaussians._xyz.data = positions
        self.gaussians._rotation.data = rotations

    def delete(self):
        """Remove selected Gaussians."""
        keep_mask = ~self.selected_mask

        self.gaussians._xyz = nn.Parameter(
            self.gaussians._xyz[keep_mask]
        )
        self.gaussians._rotation = nn.Parameter(
            self.gaussians._rotation[keep_mask]
        )
        self.gaussians._scaling = nn.Parameter(
            self.gaussians._scaling[keep_mask]
        )
        # ... update other parameters ...

        print(f"Deleted {self.selected_mask.sum()} Gaussians")
        self.selected_mask = torch.zeros(len(self.gaussians), dtype=torch.bool)
```

### Inpainting Pipeline

```python
def inpaint_removed_region(
    gaussians,
    editor,
    train_views,
    diffusion_model
):
    """Fill holes after deleting objects."""

    # Step 1: Identify disoccluded regions
    removed_mask = editor.selected_mask

    disoccluded_masks = []
    for view in train_views:
        # Render with all Gaussians
        full_render = render_gaussians(gaussians, view)

        # Render without removed Gaussians
        partial_render = render_gaussians(
            gaussians,
            view,
            skip_mask=removed_mask
        )

        # Disocluded = where partial is substantially different
        disoccluded = torch.norm(full_render - partial_render, dim=-1) > 0.1
        disoccluded_masks.append(disoccluded)

    # Step 2: Inpaint each view
    inpainted_images = []
    for view, mask in zip(train_views, disoccluded_masks):
        partial_render = render_gaussians(
            gaussians,
            view,
            skip_mask=removed_mask
        )

        # Use diffusion model to inpaint
        inpainted = diffusion_model.inpaint(
            image=partial_render,
            mask=mask,
            prompt="photorealistic scene"
        )

        inpainted_images.append(inpainted)

    # Step 3: Re-optimize Gaussians to fit inpainted views
    # Add new Gaussians in disoccluded region
    new_gaussians = initialize_gaussians_in_region(
        disoccluded_masks,
        train_views,
        num_gaussians=5000
    )

    # Optimize new Gaussians
    for iteration in range(1000):
        loss = 0
        for view, inpainted in zip(train_views, inpainted_images):
            rendered = render_gaussians(new_gaussians, view)
            loss += F.mse_loss(rendered, inpainted)

        loss.backward()
        optimizer.step()

    # Merge new Gaussians
    gaussians.merge(new_gaussians)

    return gaussians
```

## Optimization Tricks

### 1. Hierarchical Selection

```python
# Multi-level selection for efficiency
class HierarchicalGaussians:
    def __init__(self, gaussians):
        self.gaussians = gaussians
        self.build_hierarchy()

    def build_hierarchy(self):
        """Build spatial hierarchy for fast queries."""
        positions = self.gaussians.get_xyz()
        self.octree = Octree(positions, max_depth=8)

    def select_in_box(self, box_min, box_max):
        """Fast selection via octree."""
        return self.octree.query_box(box_min, box_max)
```

### 2. Differentiable Mask Rendering

```python
def render_soft_mask(gaussians, mask, camera):
    """Render with soft selection for smooth editing."""
    # Render with opacity modulated by mask
    modified_opacity = gaussians.get_opacity() * mask[:, None]

    rendered = gaussian_splatting_render(
        gaussians.get_xyz(),
        gaussians.get_rotation(),
        gaussians.get_scaling(),
        gaussians.get_color(),
        modified_opacity,
        camera
    )
    return rendered
```

### 3. Undo/Redo via State Checkpointing

```python
class EditHistory:
    def __init__(self, max_history=10):
        self.history = []
        self.current_idx = -1
        self.max_history = max_history

    def checkpoint(self, gaussians):
        """Save current state."""
        # Clear redo stack
        self.history = self.history[:self.current_idx + 1]

        # Save state
        state = {
            'positions': gaussians.get_xyz().clone(),
            'rotations': gaussians.get_rotation().clone(),
            # ... other parameters
        }

        self.history.append(state)
        self.current_idx += 1

        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.current_idx -= 1

    def undo(self, gaussians):
        """Restore previous state."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.restore_state(gaussians, self.history[self.current_idx])

    def redo(self, gaussians):
        """Restore next state."""
        if self.current_idx < len(self.history) - 1:
            self.current_idx += 1
            self.restore_state(gaussians, self.history[self.current_idx])
```

## Common Use Cases

### 1. Object Removal

```python
# Select object
editor.select_by_semantic_label("car")

# Remove
editor.delete()

# Inpaint
gaussians = inpaint_removed_region(gaussians, editor, views, diffusion_model)
```

### 2. Object Cloning

```python
# Select object
editor.select_by_click(pixel_position, camera)

# Clone
cloned_gaussians = editor.clone_selected()

# Move clone
editor.select(cloned_gaussians)
editor.translate(torch.tensor([2.0, 0.0, 0.0]))
```

### 3. Style Transfer

```python
# Select region
editor.select_in_box(box_min, box_max)

# Apply style
for iteration in range(100):
    rendered = render_selected(gaussians, editor.selected_mask, camera)
    loss = style_loss(rendered, style_target)
    loss.backward()
    optimizer.step()
```

## References

### Primary Paper

```bibtex
@article{chen2023gaussianeditor,
  title={GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting},
  author={Chen, Yiwen and Chen, Zilong and Zhang, Chi and Wang, Feng and Yang, Xiaofeng and Wang, Yikai and Cai, Zhongang and Yang, Lei and Liu, Huaping and Lin, Guosheng},
  journal={arXiv preprint arXiv:2311.14521},
  year={2023}
}
```

### Related Work

- **3D Gaussian Splatting** (Kerbl et al., 2023)
- **Segment Anything (SAM)** (Kirillov et al., 2023)
- **DreamEditor** (Zhuang et al., 2023): NeRF-based editing

---

**Next**: Try [DreamGaussian](./dream_gaussian.md) for text-driven 3D generation.

# Structured Generation

Structured generation ensures that language models produce outputs that conform to formal specifications, schemas, or grammars. This is critical for reliable API integration, data extraction, and programmatic use of LLM outputs.

## Overview

Traditional language models generate free-form text, which often requires post-processing, validation, and error handling. Structured generation constrains the generation process at decode time to guarantee valid outputs according to predefined rules.

## When to Use Structured Generation

Use structured generation when you need:

1. **API Integration**: Generating valid JSON for API calls
2. **Data Extraction**: Extracting structured information from unstructured text
3. **Code Generation**: Producing syntactically correct code
4. **Form Filling**: Generating responses that match specific formats
5. **Guaranteed Validity**: When downstream systems cannot handle malformed inputs

## Approaches

### 1. Grammar-Constrained Decoding

Uses formal grammars (context-free grammars, regular expressions) to constrain the token space at each decoding step.

**Strengths:**
- Works with any grammar (JSON, XML, SQL, Python, etc.)
- Guarantees syntactic correctness
- No model retraining required

**Weaknesses:**
- Slower inference (token masking at each step)
- May reduce output diversity
- Grammar construction can be complex

**Use when:** You need guaranteed syntactic validity for well-defined languages.

See: [grammar_constrained_decoding.md](./grammar_constrained_decoding.md)

### 2. JSON Schema Decoder

Specialized for JSON generation using finite state machines (FSM) based on JSON schemas.

**Strengths:**
- Efficient FSM-based implementation
- Native schema validation
- Handles nested structures, enums, and constraints
- Better performance than general grammar decoding

**Weaknesses:**
- Limited to JSON format
- Complex schemas can be slow
- Schema must be known in advance

**Use when:** You need to generate structured JSON data conforming to a specific schema.

See: [json_schema_decoder.md](./json_schema_decoder.md)

## Comparison Matrix

| Feature | Grammar-Constrained | JSON Schema |
|---------|-------------------|-------------|
| Output Format | Any grammar | JSON only |
| Performance | Slower (general) | Faster (specialized) |
| Flexibility | High | Medium |
| Schema Support | Manual | Native |
| Implementation Complexity | High | Medium |
| Inference Overhead | 2-5x | 1.5-3x |

## Best Practices

1. **Start Simple**: Begin with JSON schema for structured data, only use grammar decoding for special cases
2. **Cache FSMs**: Reuse FSM/grammar structures across requests with the same schema
3. **Balance Constraints**: Over-constraining can hurt output quality
4. **Test Schemas**: Validate schemas with diverse inputs before production
5. **Monitor Performance**: Structured generation adds overhead - profile your use case

## Implementation Tips

- Use logit biasing (soft constraints) when strict guarantees aren't required
- Combine with prompt engineering for better semantic quality
- Consider model size - smaller models may struggle with complex constraints
- Use temperature=0 for fully deterministic outputs

## Resources

- Outlines Library: [https://github.com/outlines-dev/outlines](https://github.com/outlines-dev/outlines)
- Guidance Library: [https://github.com/guidance-ai/guidance](https://github.com/guidance-ai/guidance)
- LMQL: [https://lmql.ai/](https://lmql.ai/)

## Example Use Cases

1. **API Client Generation**: Generate valid API requests for external services
2. **Database Queries**: Produce syntactically correct SQL queries
3. **Configuration Files**: Generate valid YAML/TOML/JSON configs
4. **Type-Safe Responses**: Ensure type safety in downstream processing
5. **Multi-step Workflows**: Chain multiple structured outputs together

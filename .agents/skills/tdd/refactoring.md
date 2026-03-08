# Refactor Candidates

After TDD cycle, look for:

- **Duplication** → Extract function/method
- **Long methods** → Break into private helpers (keep tests on public trait interface)
- **Shallow modules** → Combine or deepen (add validation, retries, error recovery)
- **Feature envy** → Move logic to where data lives
- **Primitive obsession** → Introduce newtypes (e.g., `Confidence(f32)`, `ClassId(u32)`)
- **Existing code** the new code reveals as problematic

## Rust-Specific Refactors

- Replace `.unwrap()` with `?` or graceful handling (outside tests)
- Use `thiserror` enums instead of string errors
- Prefer `impl Trait` in function args when concrete type doesn't matter
- Extract `#[cfg(test)]` mocks to keep them minimal

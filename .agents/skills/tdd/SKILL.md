---
name: tdd
description: Test-driven development with red-green-refactor loop. Use when user wants to build features or fix bugs using TDD, mentions "red-green-refactor", wants integration tests, or asks for test-first development.
---

# Test-Driven Development

## Philosophy

**Core principle**: Tests should verify behavior through public trait interfaces, not implementation details. Code can change entirely; tests shouldn't.

**Good tests** are integration-style: they exercise real code paths through trait interfaces. They describe _what_ the system does, not _how_ it does it. A good test reads like a specification — `test_cat_enter_saves_image_and_notifies` tells you exactly what capability exists. These tests survive refactors because they don't care about internal structure.

**Bad tests** are coupled to implementation. They test private methods, assert on internal state, or verify through external means (like checking the filesystem directly instead of using the trait). The warning sign: your test breaks when you refactor, but behavior hasn't changed.

See [tests.md](tests.md) for examples and [mocking.md](mocking.md) for mocking guidelines.

## Anti-Pattern: Horizontal Slices

**DO NOT write all tests first, then all implementation.** This is "horizontal slicing" — treating RED as "write all tests" and GREEN as "write all code."

This produces **crap tests**:

- Tests written in bulk test _imagined_ behavior, not _actual_ behavior
- You end up testing the _shape_ of things (struct fields, function signatures) rather than user-facing behavior
- Tests become insensitive to real changes — they pass when behavior breaks, fail when behavior is fine

**Correct approach**: Vertical slices via tracer bullets. One test → one implementation → repeat.

```
WRONG (horizontal):
  RED:   test1, test2, test3, test4, test5
  GREEN: impl1, impl2, impl3, impl4, impl5

RIGHT (vertical):
  RED→GREEN: test1→impl1
  RED→GREEN: test2→impl2
  RED→GREEN: test3→impl3
  ...
```

## Workflow

### 1. Planning

Before writing any code:

- [ ] Confirm with user what interface (trait) changes are needed
- [ ] Confirm with user which behaviors to test (prioritize)
- [ ] Identify opportunities for [deep modules](deep-modules.md)
- [ ] Design interfaces for [testability](interface-design.md)
- [ ] List the behaviors to test (not implementation steps)
- [ ] Get user approval on the plan

**You can't test everything.** Confirm with the user exactly which behaviors matter most.

### 2. Tracer Bullet

Write ONE test that confirms ONE thing:

```
RED:   Write test → cargo test --lib → test fails
GREEN: Write minimal code → cargo test --lib → test passes
```

### 3. Incremental Loop

For each remaining behavior:

```
RED:   Write next test → fails
GREEN: Minimal code to pass → passes
```

Rules:

- One test at a time
- Only enough code to pass current test
- Don't anticipate future tests
- Keep tests focused on observable behavior

### 4. Refactor

After all tests pass, look for [refactor candidates](refactoring.md):

- [ ] Extract duplication
- [ ] Deepen modules (move complexity behind simple trait interfaces)
- [ ] Replace `.unwrap()` with `?` or graceful error handling
- [ ] Run `cargo test --lib` after each refactor step
- [ ] Run `cargo clippy --all-features -- -D warnings` at the end

**Never refactor while RED.** Get to GREEN first.

## Project Conventions

- Place tests in `#[cfg(test)] mod tests` at bottom of each source file
- Use `#[tokio::test]` for async tests
- Name tests `test_<what>_<condition>_<expected>`
- Use `Mock*` structs from the same module (see [mocking.md](mocking.md))
- Use `tempfile` crate for filesystem tests
- Use `thiserror` enums for errors, not string errors

## Checklist Per Cycle

```
[ ] Test describes behavior, not implementation
[ ] Test uses public trait interface only
[ ] Test would survive internal refactor
[ ] Code is minimal for this test
[ ] No speculative features added
[ ] cargo test --lib passes
```

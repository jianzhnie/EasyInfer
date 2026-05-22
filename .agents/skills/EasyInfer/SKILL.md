```markdown
# EasyInfer Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and conventions used in the EasyInfer TypeScript codebase. You'll learn how to structure files, write and organize code, follow commit conventions, and understand the project's approach to testing. This guide is ideal for contributors seeking to maintain consistency and quality in the EasyInfer repository.

## Coding Conventions

### File Naming
- Use **snake_case** for all file names.
  - Example: `easy_infer_core.ts`, `model_utils.ts`

### Import Style
- Use **relative imports** for referencing other modules.
  - Example:
    ```typescript
    import { infer } from './inference_engine';
    ```

### Export Style
- Use **named exports** to expose functions, classes, or constants.
  - Example:
    ```typescript
    export function infer(data: DataType): ResultType { ... }
    ```

### Commit Messages
- Follow **conventional commit** style.
- Common prefixes: `refactor`, `docs`
- Keep commit messages concise (average ~56 characters).
  - Example:
    ```
    refactor: improve inference pipeline modularity
    docs: update README with usage examples
    ```

## Workflows

### Refactoring Code
**Trigger:** When improving code structure or readability without changing functionality  
**Command:** `/refactor`

1. Identify code that can be improved for clarity or maintainability.
2. Refactor the code, ensuring no change in external behavior.
3. Use named exports and relative imports as per conventions.
4. Test the refactored code.
5. Commit changes with a message starting with `refactor:`.

### Updating Documentation
**Trigger:** When adding or updating documentation files  
**Command:** `/docs`

1. Edit or create documentation files (e.g., `README.md`).
2. Ensure documentation matches the latest codebase state.
3. Commit changes with a message starting with `docs:`.

## Testing Patterns

- Test files follow the pattern: `*.test.*` (e.g., `inference_engine.test.ts`).
- The specific testing framework is **unknown**, but tests should be placed alongside or near the code they verify.
- Example test file structure:
  ```typescript
  import { infer } from './inference_engine';

  describe('infer', () => {
    it('should return expected result for valid input', () => {
      // test implementation
    });
  });
  ```

## Commands
| Command    | Purpose                                   |
|------------|-------------------------------------------|
| /refactor  | Refactor code for clarity or maintainability |
| /docs      | Update or add documentation                |
```

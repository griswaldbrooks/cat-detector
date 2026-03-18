---
name: render-diagram
description: Render a graphviz DOT file to PNG in website/static/img/
user_invocable: true
---

# Render Diagram

To render a graphviz diagram:

1. Write or update the `.dot` file
2. Render to PNG:

```bash
dot -Tpng -o website/static/img/<output_name>.png <input_file>.dot
```

Use catppuccin-mocha colors for consistency with the theme.

# Repository instructions for future prompts

- Keep every change simple, direct, and limited to what was requested.
- Do not add AI-generated filler, boilerplate, speculative features, or unnecessary abstractions.
- Do not narrate obvious Python code with comments.
- Do not use decorative comment blocks or repeated section comments such as `###`, dashed separators, or banner headings.
- Add a comment only when it explains a non-obvious reason, constraint, workaround, unit, or coordinate convention.
- Keep function and class descriptions to one line. Do not write long docstrings when a short sentence is enough.
- Prefer clear names and straightforward code over explanatory prose inside the source.
- Keep the number of files and folders as small as possible.
- Prefer editing an existing file over creating a new module, helper package, configuration layer, or script.
- Do not create extra documentation, examples, tests, utilities, or asset directories unless they are required by the requested change.
- Keep robot assets local under `fr3_assets/`; do not add external model repositories or runtime asset downloads.
- Avoid new dependencies unless the task cannot reasonably be completed with the current stack.
- Preserve the current two-table FR3 scene unless a prompt explicitly requests a scene change.
- Run the smallest relevant validation after changes and report the result concisely.

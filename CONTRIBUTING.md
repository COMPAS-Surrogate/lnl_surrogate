# Release Process

We use `python-semantic-release`.
This package will automatically determine the next version number based on the commit messages.
Example commit messages:

**Patch Release (e.g., 0.1.1):**
```
fix: resolve issue with missing argument
```

**Minor Release (e.g., 0.2.0):**
```
feat: add new feature to API
```

**Major Release (e.g., 1.0.0):**
```
    feat!: introduce breaking change to architecture
```
The ! after feat indicates a breaking change.
If your commit messages don't match these patterns, python-semantic-release will decide that no release is needed (no_release).

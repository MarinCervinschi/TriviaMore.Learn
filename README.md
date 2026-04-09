# TriviaMore Learn

Static multi-course study site built with Astro, MDX, and Tailwind CSS. First course: **Deep Learning** (15 chapters).

Content is rendered from MDX with KaTeX math formulas, navigable sidebar, table of contents, and full-text search via Pagefind.

## Getting started

```bash
pnpm install
pnpm dev        # http://localhost:4321
```

## Build & preview

```bash
pnpm build      # builds static site + Pagefind index
pnpm preview    # serves dist/
```

## Content migration

Source markdown files (Notion exports) live in `Deep Learning/Teory/`. To re-run the migration:

```bash
node scripts/migrate-notion.mjs
```

This generates MDX files in `src/content/courses/deep-learning/` and copies images to `public/images/deep-learning/`.

## Adding a new course

1. Create `src/content/courses/<slug>/` with a `_meta.json`:
   ```json
   { "title": "Course Name", "slug": "course-slug", "description": "...", "chapters": 5 }
   ```
2. Add `.mdx` files with frontmatter: `title`, `order`, `course`
3. Put images in `public/images/<slug>/`

No routing or component changes needed.

## Stack

- [Astro](https://astro.build) v6 + MDX
- [Tailwind CSS](https://tailwindcss.com) v4
- [KaTeX](https://katex.org) (remark-math + rehype-katex)
- [Pagefind](https://pagefind.app) for search

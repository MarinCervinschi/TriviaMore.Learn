import fs from 'node:fs';
import path from 'node:path';

export interface CourseMeta {
  title: string;
  slug: string;
  description: string;
  chapters: number;
}

export function getCourseMeta(courseSlug: string): CourseMeta {
  const metaPath = path.join(
    process.cwd(), 'src/content/courses', courseSlug, '_meta.json'
  );
  return JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
}

/** Extract filename slug from entry.id (e.g., "deep-learning/01-deep-neural-networks" → "01-deep-neural-networks") */
export function getSlugFromId(entryId: string): string {
  const parts = entryId.split('/');
  return parts[parts.length - 1];
}

export function getAllCourseMetas(): CourseMeta[] {
  const coursesDir = path.join(process.cwd(), 'src/content/courses');
  return fs.readdirSync(coursesDir, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => getCourseMeta(d.name));
}

import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { glob } from 'astro/loaders';

const courses = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/courses' }),
  schema: z.object({
    title: z.string(),
    order: z.number(),
    description: z.string().optional(),
    course: z.string(),
  }),
});

export const collections = { courses };

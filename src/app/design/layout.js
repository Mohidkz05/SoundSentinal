import { notFound } from 'next/navigation';

/**
 * /design is the living design-system reference, for whoever builds the
 * product, not for the person checking a clip. It runs under `npm run dev`
 * and is a 404 in every production build, so it cannot be reached on the
 * launched site even by typing the URL.
 */
export default function DesignLayout({ children }) {
  if (process.env.NODE_ENV === 'production') notFound();
  return children;
}

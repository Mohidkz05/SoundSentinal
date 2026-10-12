'use client';

import React from 'react';
import Link from 'next/link';
import { cva } from 'class-variance-authority';
import { Slot } from 'radix-ui';
import { cn } from '@/lib/utils';

/**
 * Buttons — shadcn/ui's Button (cva variants + Radix Slot), themed with this
 * design system's tokens rather than shadcn's zinc defaults (10 October 2026).
 *
 * The rule is still one `primary` per screen: if two things are primary,
 * neither is. `quiet` is the default for anything that isn't the point.
 *
 * Pass `href` and the same control renders as a Next link through Slot, so
 * navigation is an <a> and an action is a <button> — never one wrapped in
 * the other (invalid nesting, two tab stops, a "button" that navigates).
 *
 * Elevation carries meaning here: the primary action sits on a brand-tinted
 * shadow and lifts a pixel on hover; pressing it presses it back down. The
 * focus ring is the global `:focus-visible` outline in the accent colour.
 *
 * Sizes draw at 36 / 44 / 52px, and every size has at least a 44px hit area:
 * `sm` extends its target with an invisible ::after.
 */
export const buttonVariants = cva(
  [
    'relative inline-flex shrink-0 select-none items-center justify-center gap-2 whitespace-nowrap',
    'rounded-[var(--radius-md)] border font-sans font-semibold no-underline',
    'transition-[background-color,border-color,color,box-shadow,translate] duration-[var(--duration-fast)] ease-[var(--ease-instrument)]',
    'active:translate-y-px disabled:pointer-events-none disabled:opacity-45 aria-disabled:pointer-events-none aria-disabled:opacity-45',
    "[&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  ].join(' '),
  {
    variants: {
      variant: {
        // The one action that moves the user forward.
        primary:
          'border-transparent bg-accent text-accent-contrast shadow-[var(--shadow-action)] hover:-translate-y-px hover:bg-accent-hover hover:shadow-[var(--shadow-action-hover)] active:translate-y-0 active:shadow-[var(--shadow-action)]',
        // Equal-weight alternatives, and anything on a panel.
        secondary:
          'border-line-strong bg-raised text-primary shadow-[var(--shadow-panel)] hover:border-accent hover:text-accent',
        // Tertiary actions: back, cancel, change file.
        quiet: 'border-transparent bg-transparent text-secondary hover:bg-overlay hover:text-primary',
        // Destructive or dismissive. The danger token, never a verdict orange.
        danger: 'border-danger bg-transparent text-danger hover:bg-danger hover:text-canvas',
        link: 'h-auto border-transparent px-0 text-accent underline-offset-4 hover:underline',
      },
      size: {
        sm: 'h-9 px-3.5 text-small after:absolute after:inset-x-0 after:-inset-y-1 after:content-[""]',
        md: 'h-11 px-5 text-body',
        lg: 'h-[3.25rem] px-7 text-body',
      },
    },
    defaultVariants: { variant: 'secondary', size: 'md' },
  }
);

/** Kept for callers that style a non-button element as one. */
export function buttonClasses({ variant = 'secondary', size = 'md', className = '' } = {}) {
  return cn(buttonVariants({ variant, size }), className);
}

export function Button({
  variant = 'secondary',
  size = 'md',
  className = '',
  loading = false,
  href,
  asChild = false,
  children,
  ...props
}) {
  /* A loading button is disabled (so a click can't land twice) but must not
     look disabled: it is the thing the user is waiting on. */
  const classes = cn(buttonVariants({ variant, size }), loading && 'cursor-progress opacity-100!', className);

  if (href) {
    return (
      <Slot.Root data-slot="button" className={classes}>
        <Link href={href} {...props}>
          {children}
        </Link>
      </Slot.Root>
    );
  }

  const Comp = asChild ? Slot.Root : 'button';
  return (
    <Comp
      data-slot="button"
      type={asChild ? undefined : 'button'}
      {...props}
      className={classes}
      aria-busy={loading || undefined}
      /* After the spread: a loading button is disabled whatever else was
         passed, so the click cannot land twice. */
      disabled={asChild ? undefined : loading || props.disabled}
    >
      {loading && <Spinner />}
      {children}
    </Comp>
  );
}

function Spinner() {
  return (
    <svg className="h-4 w-4 animate-spin" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2" />
      <path d="M14.5 8A6.5 6.5 0 0 0 8 1.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

import React from 'react';
import { CircleAlert, Info } from 'lucide-react';
import { Alert, AlertDescription } from '@components/ui/alert';
import { cn } from '@/lib/utils';

/**
 * A sentence the user has to notice — shadcn/ui's Alert with two tones:
 *
 *   tone="danger"  a system failure (rejected file, server down). role=alert,
 *                  the danger token in a bordered box — never a verdict orange.
 *   tone="caveat"  a qualification of a result or preview. Muted, no box, an
 *                  info icon; it must not look like an error.
 *
 * Icon plus text, never colour alone. Errors say what happened and what to do.
 */
export function Notice({ tone = 'caveat', children, className = '' }) {
  const danger = tone === 'danger';
  const Icon = danger ? CircleAlert : Info;
  return (
    <Alert
      variant={danger ? 'danger' : 'caveat'}
      role={danger ? 'alert' : 'note'}
      className={cn('max-w-[72ch]', className)}
    >
      <Icon aria-hidden="true" strokeWidth={1.8} />
      <AlertDescription className={cn('text-small', danger ? 'text-danger' : 'text-muted')}>
        <p>{children}</p>
      </AlertDescription>
    </Alert>
  );
}

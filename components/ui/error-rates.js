import React from 'react';
import { formatPercent } from '../../src/lib/verdict';

/**
 * How often the served model is wrong, per test set: genuine recordings it
 * flags and fakes it lets through, at the threshold it is actually served at.
 *
 * The rows come from the API (app.py reads evaluate.py's reports beside the
 * checkpoint and drops any measured at another threshold), never from this
 * file — so the same table on / and /result always describes the live model.
 * Two columns rather than one "accuracy": the two mistakes cost different
 * things, and folding them together is how detectors end up quoting a number
 * nobody can act on.
 */
function Figure({ label, value, of }) {
  return (
    <div className="flex flex-col gap-[var(--space-tight)]">
      <span className="tick-label">{label}</span>
      <span className="tabular text-h3 leading-none text-primary">{formatPercent(value)}</span>
      <span className="tabular text-small text-muted">of {of.toLocaleString('en-GB')}</span>
    </div>
  );
}

export function ErrorRates({ measured }) {
  return (
    <>
      {/* Phones: one block per test set, the two figures side by side. A
          three-column table at 390px scrolled sideways and cut both figures
          off — the two numbers the table exists for. */}
      <ul className="flex flex-col sm:hidden">
        {measured.map((m) => (
          <li key={m.set} className="flex flex-col gap-[var(--space-stack)] border-b border-line py-5 first:pt-0 last:border-0">
            <div>
              <p className="font-semibold text-primary">{m.set}</p>
              {m.about && <p className="mt-1 text-small text-muted">{m.about}</p>}
            </div>
            <div className="grid grid-cols-2 gap-4">
              <Figure label="Real flagged" value={m.real_flagged} of={m.n_real} />
              <Figure label="Fakes missed" value={m.fakes_passed} of={m.n_fake} />
            </div>
          </li>
        ))}
      </ul>

    <div className="hidden overflow-x-auto sm:block">
      <table className="data-table">
        <thead>
          <tr>
            <th scope="col">Tested on</th>
            <th scope="col">Real recordings flagged</th>
            <th scope="col">Fakes missed</th>
          </tr>
        </thead>
        <tbody>
          {measured.map((m) => (
            <tr key={m.set}>
              <th scope="row" className="max-w-[38ch] font-normal">
                <span className="block font-semibold text-primary">{m.set}</span>
                {m.about && <span className="mt-1 block text-small text-muted">{m.about}</span>}
              </th>
              <td>
                <span className="tabular block text-h3 leading-none text-primary">
                  {formatPercent(m.real_flagged)}
                </span>
                <span className="tabular mt-2 block text-small text-muted">
                  of {m.n_real.toLocaleString('en-GB')}
                </span>
              </td>
              <td>
                <span className="tabular block text-h3 leading-none text-primary">
                  {formatPercent(m.fakes_passed)}
                </span>
                <span className="tabular mt-2 block text-small text-muted">
                  of {m.n_fake.toLocaleString('en-GB')}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
    </>
  );
}

import React from 'react';

export default function Input({ error, className = '', ...props }) {
  const base = 'block w-full rounded-md border bg-surface px-3 py-2 focus:outline-none focus:ring-2';
  const ring = error ? 'border-error focus:ring-error' : 'border-border focus:ring-primary';
  return (
    <>
      <input className={`${base} ${ring} ${className}`} {...props} />
      {error && <p className="mt-1 text-xs text-error">{error}</p>}
    </>
  );
}

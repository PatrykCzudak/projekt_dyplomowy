import React from 'react';
import Spinner from './Spinner';

export default function Button({
  children,
  variant = 'primary',  // 'primary' | 'secondary'
  disabled = false,
  loading = false,
  className = '',
  ...props
}) {
  const base = 'inline-flex items-center justify-center font-medium rounded-md transition';
  const variants = {
    primary:  'bg-primary text-white hover:bg-primary-light disabled:bg-primary-dark',
    secondary:'bg-secondary text-white hover:bg-secondary-light disabled:bg-secondary-dark',
  };
  return (
    <button
      className={`${base} ${variants[variant]} px-4 py-2 disabled:opacity-50 ${className}`}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <Spinner className="w-4 h-4 mr-2" />}
      {children}
    </button>
  );
}

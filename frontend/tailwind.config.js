module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#1E40AF',
          light:   '#3B82F6',
          dark:    '#1E3A8A',
        },
        secondary: {
          DEFAULT: '#9333EA',
          light:   '#C084FC',
          dark:    '#6B21A8',
        },
        success:   '#10B981',
        error:     '#EF4444',
        warning:   '#F59E0B',
        info:      '#3B82F6',
        background: '#F3F4F6',
        surface:    '#FFFFFF',
        border:     '#D1D5DB',
      },
      spacing: {
        1: '4px',
        2: '8px',
        3: '12px',
        4: '16px',
        5: '20px',
        6: '24px',
        8: '32px',
        10: '40px',
      },
      fontSize: {
        xs: ['0.75rem', { lineHeight: '1rem' }],
        sm: ['0.875rem', { lineHeight: '1.25rem' }],
        base: ['1rem', { lineHeight: '1.5rem' }],
        lg: ['1.125rem', { lineHeight: '1.75rem' }],
        xl: ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
      },
    },
  },
  plugins: [],
};

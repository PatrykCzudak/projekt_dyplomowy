module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#1E40AF',    // Tailwind blue-800
          light:   '#3B82F6',    // blue-500
          dark:    '#1E3A8A',    // blue-900
        },
        secondary: {
          DEFAULT: '#9333EA',    // purple-600
          light:   '#C084FC',    // purple-300
          dark:    '#6B21A8',    // purple-800
        },
        success:   '#10B981',      // green-500
        error:     '#EF4444',      // red-500
        warning:   '#F59E0B',      // yellow-500
        info:      '#3B82F6',      // blue-500
        background: '#F3F4F6',     // gray-100
        surface:    '#FFFFFF',     // white
        border:     '#D1D5DB',     // gray-300
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

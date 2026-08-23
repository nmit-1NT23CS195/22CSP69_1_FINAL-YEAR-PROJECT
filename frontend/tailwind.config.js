/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        zinc: {
          950: '#09090b',
          900: '#18181b',
        }
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      animation: {
        'fade-up': 'fade-up 0.6s ease-out forwards',
        'float-particle': 'float-particle 5s ease-in-out infinite',
        'glow-pulse': 'glow-pulse 2.5s ease-in-out infinite',
        'border-pulse': 'border-pulse 3s ease-in-out infinite',
        'pulse-slow': 'pulse-slow 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(16px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'float-particle': {
          '0%, 100%': { transform: 'translateY(0) scale(1)', opacity: '0.4' },
          '33%': { transform: 'translateY(-8px) scale(1.2)', opacity: '0.6' },
          '66%': { transform: 'translateY(4px) scale(0.9)', opacity: '0.3' },
        },
        'glow-pulse': {
          '0%, 100%': { boxShadow: '0 0 15px rgba(99,102,241,0.3)' },
          '50%': { boxShadow: '0 0 30px rgba(99,102,241,0.6), 0 0 60px rgba(168,85,247,0.2)' },
        },
        'border-pulse': {
          '0%, 100%': { borderColor: 'rgba(99,102,241,0.2)' },
          '50%': { borderColor: 'rgba(99,102,241,0.5)' },
        },
        'pulse-slow': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.5' },
        },
      },
      backdropBlur: {
        '2xl': '40px',
        '3xl': '64px',
      },
    },
  },
  plugins: [],
}

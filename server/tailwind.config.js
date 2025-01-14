/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        titleColor: '#67f5a0',
      },
      fontFamily: {
        custom: ["Xanh Mono", "sans-serif"],
      }
    },
  },
  plugins: [],
}
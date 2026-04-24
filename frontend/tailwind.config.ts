import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f0f4ff",
          100: "#dce6ff",
          500: "#3b5bdb",
          600: "#2f4ac7",
          900: "#1a2a6c",
        },
      },
    },
  },
  plugins: [],
};

export default config;

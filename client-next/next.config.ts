import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /** Pin workspace root when dev is started from this directory (avoids stray parent lockfile warnings). */
  turbopack: {
    root: process.cwd(),
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_SERVER_URL ?? "http://localhost:8001"}/:path*`,
      },
    ];
  },
};

export default nextConfig;

import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Rajdhani, Share_Tech_Mono } from "next/font/google";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const rajdhani = Rajdhani({
  variable: "--font-rajdhani-face",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

const shareTechMono = Share_Tech_Mono({
  variable: "--font-stm-face",
  subsets: ["latin"],
  weight: "400",
});

export const metadata: Metadata = {
  title: "OPERATION: VISUALREF — Visual Intelligence Command",
  description:
    "Classified visual intelligence retrieval and feedback system. Mission-grade image analysis powered by AI.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${rajdhani.variable} ${shareTechMono.variable} antialiased font-sans`}
      >
        <TooltipProvider>{children}</TooltipProvider>
      </body>
    </html>
  );
}

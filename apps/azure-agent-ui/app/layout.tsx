import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Azure Agent UI",
  description: "Web chat client for Azure Agent",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}

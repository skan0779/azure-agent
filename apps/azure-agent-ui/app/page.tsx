"use client";

import { AuthProvider } from "@/app/auth-provider";
import { ChatShell } from "@/app/chat-shell";

export default function Home() {
  return (
    <AuthProvider>
      <ChatShell />
    </AuthProvider>
  );
}

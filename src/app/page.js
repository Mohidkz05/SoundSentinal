'use client';

import React from 'react';
import Image from "next/image";
import Header from "../../components/header";
import { useTheme } from "../../components/theme";

export default function Home() {
  const { darkMode } = useTheme();
  return (
    <div className="flex min-h-screen flex-col items-center p-8 bg-white dark:bg-black">
      <Header />
      <main className="flex flex-col items-center justify-center flex-1 w-full">
        <h1 className="text-4xl sm:text-6xl font-bold mb-4">Sound Sentinel</h1>
        <button className="bg-neutral-800 text-white font-medium py-3 px-6 rounded-full hover:bg-neutral-900 transition-colors">
          Get Started
        </button>
      </main>
    </div>
  );
}

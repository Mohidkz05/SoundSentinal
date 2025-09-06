'use client';

import React from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { useTheme } from './theme'; // Corrected import path


export default function Header() {
  const { darkMode, toggleDarkMode } = useTheme();

  return (
    <header className="w-full py-4 px-8 border-b border-gray-300 dark:border-gray-700 bg-white dark:bg-black flex items-center justify-between">
      <div className="flex items-center">
        {/* Replace with your logo SVG in the public folder */}
        <div className="w-10 h-10 bg-gray-300 dark:bg-gray-700 rounded-full flex items-center justify-center">
          <span className="text-xs font-bold text-gray-700 dark:text-gray-300">LOGO</span>
        </div>
        <span className="ml-2 font-bold text-lg">SOUND SENTINEL</span>
      </div>
      <nav className="hidden md:flex items-center space-x-4">
        <Link href="/" className="text-sm hover:text-gray-600 dark:hover:text-gray-400">Home</Link>
        <Link href="upload" className="text-sm hover:text-gray-600 dark:hover:text-gray-400">Upload</Link>
        <Link href="result" className="text-sm hover:text-gray-600 dark:hover:text-gray-400">Result</Link>
        <Link href="#" className="text-sm hover:text-gray-600 dark:hover:text-gray-400">Resources</Link>
        <Link href="#" className="text-sm hover:text-gray-600 dark:hover:text-gray-400">Contact</Link>
        <Link href="#" className="text-sm hover:text-gray-600 dark:hover:text-gray-400">Link</Link>
        <Link href="#" className="bg-black text-white px-4 py-2 rounded-full text-sm">Home</Link>
      </nav>
      {/* Dark/Light mode toggle button */}
      <button 
        onClick={toggleDarkMode} 
        className="p-2 ml-4 rounded-full bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200"
      >
        {darkMode ? (
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
            <path d="M12 2.25a.75.75 0 01.75.75v2.25a.75.75 0 01-1.5 0V3a.75.75 0 01.75-.75zM7.5 12a4.5 4.5 0 119 0 4.5 4.5 0 01-9 0zM18.894 6.106a.75.75 0 00-1.06-1.06l-1.591 1.59a.75.75 0 101.06 1.06l1.59-1.591zM10.755 20.871a.75.75 0 00-.75-.75h-2.25a.75.75 0 000 1.5h2.25a.75.75 0 00.75-.75zM5.106 18.894a.75.75 0 001.06-1.06l-1.59-1.591a.75.75 0 10-1.06 1.06l1.59 1.591zM10.755 3.129a.75.75 0 00-.75.75v2.25a.75.75 0 001.5 0V3.879a.75.75 0 00-.75-.75zM21.75 12a.75.75 0 01-.75.75h-2.25a.75.75 0 010-1.5h2.25a.75.75 0 01.75.75zM17.245 20.871a.75.75 0 001.5 0v-2.25a.75.75 0 00-1.5 0v2.25zM18.894 17.444a.75.75 0 00-1.06-1.06l-1.591 1.59a.75.75 0 101.06 1.06l1.59-1.591zM12 15.75a3.75 3.75 0 100-7.5 3.75 3.75 0 000 7.5z" clipRule="evenodd" />
          </svg>
        ) : (
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
            <path fillRule="evenodd" d="M9.528 1.718a.75.75 0 01.976.233l2.846 6.945a.75.75 0 01-1.372.562l-2.029-4.965-2.029 4.965a.75.75 0 01-1.372-.562L8.552 1.951a.75.75 0 01.976-.233zM7.5 12a4.5 4.5 0 119 0 4.5 4.5 0 01-9 0z" clipRule="evenodd" />
            <path d="M11.516 2.016a.75.75 0 01.734.783c-.78 7.33-.426 14.881 2.531 16.518-.756.248-1.545.37-2.351.37-6.666 0-12.062-5.396-12.062-12.062 0-2.344.673-4.505 1.834-6.32a.75.75 0 01.711.168.75.75 0 01.168.711A10.518 10.518 0 0012 21.016c.806 0 1.595-.122 2.351-.371-2.957-1.637-3.311-9.188-2.531-16.518a.75.75 0 01.734-.783z" />
          </svg>
        )}
      </button>
    </header>
  );
}

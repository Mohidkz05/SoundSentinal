'use client';

import React, { createContext, useState, useEffect, useContext } from 'react';

// Create a context to hold the theme state and toggle function
export const ThemeContext = createContext();

// Create the provider component that will wrap the entire app
export function Theme({ children }) {
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    // Check for user's system preference on initial load
    const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setDarkMode(isDark);
    if (isDark) {
      document.documentElement.classList.add('dark');
    }
  }, []);

  const toggleDarkMode = () => {
    setDarkMode(prevMode => {
      const newMode = !prevMode;
      if (newMode) {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
      return newMode;
    });
  };

  return (
    <ThemeContext.Provider value={{ darkMode, toggleDarkMode }}>
      {children}
    </ThemeContext.Provider>
  );
}

// Custom hook to easily use the theme context
export const useTheme = () => useContext(ThemeContext);

'use client';

import React, { createContext, useState, useEffect, useContext, useCallback } from 'react';

const STORAGE_KEY = 'soundsentinal-theme';

export const ThemeContext = createContext(null);

export function Theme({ children }) {
  // The inline script in layout.js has already put the right class on <html>
  // before paint. Read back from that rather than guessing, so the first render
  // agrees with what is on screen.
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    setDarkMode(document.documentElement.classList.contains('dark'));
  }, []);

  const toggleDarkMode = useCallback(() => {
    setDarkMode((prev) => {
      const next = !prev;
      document.documentElement.classList.toggle('dark', next);
      try {
        localStorage.setItem(STORAGE_KEY, next ? 'dark' : 'light');
      } catch {
        // Private mode or storage disabled — the toggle still works for this
        // session, it just won't be remembered.
      }
      return next;
    });
  }, []);

  return (
    <ThemeContext.Provider value={{ darkMode, toggleDarkMode }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => useContext(ThemeContext);

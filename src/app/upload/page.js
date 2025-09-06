'use client';

import React, { useState, useCallback } from 'react';
import Image from 'next/image';
import Header from "../../../components/header";

export default function UploadPage() {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState(null);

  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true);
    }
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
      e.dataTransfer.clearData();
    }
  }, []);

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center p-8 bg-white dark:bg-black">
      <Header />
      <main className="flex flex-col items-center justify-start flex-1 w-full max-w-2xl py-12">
        <h1 className="text-4xl sm:text-5xl font-bold mb-4 text-center">
          Upload Audio
        </h1>
        <p className="text-sm text-gray-600 dark:text-gray-400 text-center mb-8">
          Please upload your choice of audio. Note that only files up to 5mb will be accepted
        </p>

        <div
          className={`w-full border-2 border-dashed rounded-lg p-8 flex flex-col items-center justify-center transition-colors duration-200 ${
            isDragging ? 'border-blue-500 bg-blue-50 dark:bg-blue-950' : 'border-gray-300 dark:border-gray-700'
          }`}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          {file ? (
            <p className="text-lg font-semibold text-gray-800 dark:text-gray-200">
              File selected: {file.name}
            </p>
          ) : (
            <>
              <div className="mb-4">
                <svg className={`mx-auto h-12 w-12 text-gray-400 dark:text-gray-600 ${isDragging ? 'text-blue-500' : ''}`} stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                  <path d="M28 8H12a2 2 0 00-2 2v15.28a2 2 0 00.5.54L14.46 29H36a2 2 0 002-2V10a2 2 0 00-2-2z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M14 28V16a2 2 0 012-2h12a2 2 0 012 2v12" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M24 16v12M18 22h12" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
                <span className="font-semibold">Drag and drop audio here</span> or{' '}
                <label htmlFor="file-upload" className="relative cursor-pointer font-medium text-blue-600 dark:text-blue-400 hover:text-blue-500">
                  <span className="text-sm">click to upload</span>
                  <input id="file-upload" name="file-upload" type="file" className="sr-only" onChange={handleFileSelect} />
                </label>
              </p>
            </>
          )}
        </div>

        <div className="mt-12 flex flex-col items-center">
          <h2 className="text-lg font-semibold mb-4">Accepted Formats</h2>
          <div className="flex space-x-8">
            <h1>MP3, Wav</h1>
            {/* <div className="flex flex-col items-center">
              <Image src="/mp3-icon.svg" alt="MP3 icon" width={64} height={64} />
              <span className="mt-2 text-sm font-semibold">MP3</span>
            </div>
            <div className="flex flex-col items-center">
              <Image src="/wav-icon.svg" alt="WAV icon" width={64} height={64} />
              <span className="mt-2 text-sm font-semibold">WAV</span>
            </div> */}
          </div>
        </div>
      </main>
    </div>
  );
}

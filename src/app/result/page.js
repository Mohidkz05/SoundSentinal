'use client';

import Header from '../../../components/header';
import React, { useState } from 'react';
import Image from 'next/image';



const resultsData = {
  unlikely: {
    colorClass: 'text-green-500',
    description: 'This audio is Unlikely to have been AI generated.',
    barColor: 'bg-green-500'
  },
  possibly: {
    colorClass: 'text-yellow-500',
    description: 'We cannot be completely certain but there is a chance that this audio is AI generated.',
    barColor: 'bg-yellow-500'
  },
  likely: {
    colorClass: 'text-orange-500',
    description: 'This audio is Likely to have been AI generated.',
    barColor: 'bg-orange-500'
  },
  veryLikely: {
    colorClass: 'text-red-500',
    description: 'This audio is Very Likely to have been AI generated.',
    barColor: 'bg-red-500'
  }
};

export default function ResultPage() {
  const [result, setResult] = useState('unlikely');


  const currentResult = resultsData[result];


  return (
    <div className="flex min-h-screen flex-col items-center bg-white dark:bg-black font-sans">
      <Header />
      <main className="flex flex-col items-center justify-start flex-1 w-full py-12 px-4 sm:px-8">
        <div className="w-full max-w-2xl text-center">
          <h1 className="text-4xl sm:text-5xl font-bold mb-8">
            Result: <span className={currentResult.colorClass}>{result.charAt(0).toUpperCase() + result.slice(1).replace(/([A-Z])/g, ' $1')}</span>
          </h1>

          <p className="text-sm sm:text-lg text-gray-600 dark:text-gray-400 mb-8 max-w-lg mx-auto">
            {currentResult.description}
          </p>

          {/* This is the horizontal line and progress bar */}
          <div className="w-full h-1 bg-gray-200 dark:bg-gray-700 relative mb-8">
            <div className={`absolute left-0 h-full ${currentResult.barColor} transition-all duration-500 ease-in-out`} style={{
              width: result === 'unlikely' ? '25%' :
                     result === 'possibly' ? '50%' :
                     result === 'likely' ? '75%' : '100%'
            }}></div>
          </div>
          
          {/* This is a button to display the results in your preview */}
          <button className="bg-neutral-800 text-white font-medium py-3 px-6 rounded-full hover:bg-neutral-900 transition-colors">
            Analyze another audio
          </button>
        </div>

        {/* This section is for demonstration purposes only. You can remove it. */}
        <div className="mt-12 flex flex-wrap justify-center gap-4">
          <button
            className="bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 px-4 py-2 rounded-full text-sm hover:bg-gray-200 dark:hover:bg-gray-700"
            onClick={() => setResult('unlikely')}
          >
            Show Unlikely
          </button>
          <button
            className="bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 px-4 py-2 rounded-full text-sm hover:bg-gray-200 dark:hover:bg-gray-700"
            onClick={() => setResult('possibly')}
          >
            Show Possibly
          </button>
          <button
            className="bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 px-4 py-2 rounded-full text-sm hover:bg-gray-200 dark:hover:bg-gray-700"
            onClick={() => setResult('likely')}
          >
            Show Likely
          </button>
          <button
            className="bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 px-4 py-2 rounded-full text-sm hover:bg-gray-200 dark:hover:bg-gray-700"
            onClick={() => setResult('veryLikely')}
          >
            Show Very Likely
          </button>
        </div>
      </main>
    </div>
  );
}

import { useState } from 'react';
import { CogniFlowSidebar } from './components/CogniFlowSidebar';

export default function App() {
  return (
    <div className="size-full bg-[#0a0a0a]" style={{ fontFamily: 'Inter, sans-serif' }}>
      {/* Main content area - simulated workspace */}
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-gray-600 text-lg">Your Workspace</div>
      </div>
      
      <CogniFlowSidebar />
    </div>
  );
}
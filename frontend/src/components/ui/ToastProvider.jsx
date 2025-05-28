import React, { createContext, useContext, useState } from 'react';

const ToastContext = createContext();
export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);
  const add = (message, type = 'info') => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => setToasts((prev) => prev.filter(t => t.id !== id)), 3000);
  };
  return (
    <ToastContext.Provider value={add}>
      {children}
      <div className="fixed bottom-4 right-4 space-y-2">
        {toasts.map(({ id, message, type }) => (
          <div key={id}
               className={`px-4 py-2 rounded shadow ${{
                 info: 'bg-info text-white',
                 success: 'bg-success text-white',
                 error: 'bg-error text-white',
               }[type]}`}>
            {message}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
export const useToast = () => useContext(ToastContext);

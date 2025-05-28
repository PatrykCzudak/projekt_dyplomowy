import { User } from 'lucide-react';

export default function Navbar() {
  return (
    <header className="w-full bg-gray-950 border-b border-gray-800 px-4 py-3 flex items-center justify-between">
      <h1 className="text-xl font-semibold">Menu</h1>
      {/*Ikona profilu użytkownika po prawej stronie*/}
      <div className="flex items-center space-x-2">
        <User className="w-5 h-5 text-gray-400" />
        <span className="hidden sm:inline text-sm text-gray-300">Użytkownik</span>
      </div>
    </header>
  );
}

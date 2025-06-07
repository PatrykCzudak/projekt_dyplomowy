import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Navbar from './Navbar';

export default function Layout() {
  return (
    <div className="min-h-screen flex bg-gray-900 text-gray-100">
      {/* sidebar nawigacyjny */}
      <Sidebar />
      {/* top navbar + dynamiczna zawartość strony */}
      <div className="flex-1 flex flex-col">
        <Navbar />
        <main className="flex-1 p-4">
          <Outlet /> 
        </main>
      </div>
    </div>
  );
}

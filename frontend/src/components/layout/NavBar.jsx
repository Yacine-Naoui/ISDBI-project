import { Bell, Search, Menu, Moon } from "lucide-react";

const NavBar = ({ toggleSidebar, hasNewNotifications }) => {
  return (
    <nav className="w-full flex justify-between items-center px-4 py-2 bg-white shadow">
      {/* Left: Hamburger + App Name */}
      <div className="flex items-center gap-3">
        <button onClick={toggleSidebar} className="text-gray-700 cursor-pointer">
          <Menu />
        </button>
        <span className="font-semibold text-lg text-gray-800">Al Murshid</span>
      </div>

      {/* Right: Search, Notifications, Theme Placeholder */}
      <div className="flex items-center gap-4">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={16} />
          <input
            type="text"
            placeholder="Search standards"
            className="pl-9 pr-3 py-1.5 rounded-full bg-gray-100 text-sm text-gray-700 focus:outline-none"
          />
        </div>

        {/* Notifications */}
        <div className="relative flex items-center">
          <button className="relative text-gray-700">
            <Bell />
            {hasNewNotifications && (
              <span className="absolute top-0 right-0 block h-2 w-2 rounded-full bg-red-500"></span>
            )}
          </button>
        </div>

        {/* Theme toggle placeholder (inactive) */}
        <button className="text-gray-700 cursor-not-allowed opacity-50">
          <Moon />
        </button>
      </div>
    </nav>
  );
};

export default NavBar;

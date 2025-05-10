import {
  X,
  MessageSquare,
  BookOpen,
  FileText,
  Search,
  Sparkles,
} from "lucide-react";
import { Link } from "react-router-dom";

const Sidebar = ({ isOpen, toggleSidebar }) => {
  return (
    <>
      {/* Overlay */}
      <div
        className={`fixed inset-0 bg-black bg-opacity-40 transition-opacity duration-300 z-30 ${
          isOpen ? "opacity-40 visible" : "opacity-0 invisible"
        }`}
        onClick={toggleSidebar}
      />

      {/* Sidebar */}
      <div
        className={`fixed top-0 left-0 h-full w-64 bg-white shadow-md z-40 transform transition-transform duration-300 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        } flex flex-col`}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gray-300" />
            <div>
              <h4 className="font-semibold text-sm text-gray-800">Bank User</h4>
              <p className="text-xs text-gray-500">Admin</p>
            </div>
          </div>
          <button
            onClick={toggleSidebar}
            className="text-gray-600 cursor-pointer"
          >
            <X />
          </button>
        </div>

        {/* Navigation */}
        <div className="px-4 py-6 space-y-4 basis-[50%]">
          <h3 className="text-xs text-gray-500 tracking-wide uppercase">
            Navigation
          </h3>
          <nav className="space-y-2">
            <Link
              to={"/report"}
              className="w-full flex items-center gap-3 text-gray-700 hover:bg-blue-50 px-3 py-3 rounded-md transition"
            >
              <MessageSquare size={18} />
              <span className="text-sm font-medium">Chat Interface</span>
            </Link>

            <Link
              to={"/report3"}
              className="w-full flex items-center gap-3 text-gray-700 hover:bg-blue-50 px-3 py-3 rounded-md transition"
            >
              <FileText size={18} />
              <span className="text-sm font-medium">Entries Journalizer</span>
            </Link>

            <Link
              to={"/report2"} // You can change this route later
              className="w-full flex items-center gap-3 text-gray-700 hover:bg-blue-50 px-3 py-3 rounded-md transition"
            >
              <Search size={18} />
              <span className="text-sm font-medium">FAS Identifier</span>
            </Link>

            <Link
              to={"/report4"} // You can change this route later
              className="w-full flex items-center gap-3 text-gray-700 hover:bg-blue-50 px-3 py-3 rounded-md transition"
            >
              <Sparkles size={18} />
              <span className="text-sm font-medium">FAS Enhancer</span>
            </Link>

            <Link
              to={"/stand"}
              className="w-full flex items-center gap-3 text-gray-700 hover:bg-blue-50 px-3 py-3 rounded-md transition"
            >
              <BookOpen size={18} />
              <span className="text-sm font-medium">FAS Standards</span>
            </Link>
          </nav>
        </div>

        {/* Recent Cases () */}
        <div className=" px-4 py-6">
          <h5 className="text-xs text-gray-500 mb-3 uppercase font-semibold">
            Recent Cases
          </h5>
          <ul className="text-sm space-y-6">
            <li className="flex items-center gap-2 text-gray-700">
              <span className="w-2 h-2 rounded-full bg-green-500" />
              Murabaha Financing
            </li>
            <li className="flex items-center gap-2 text-gray-700">
              <span className="w-2 h-2 rounded-full bg-red-500" />
              Ijara Asset Lease
            </li>
            <li className="flex items-center gap-2 text-gray-700">
              <span className="w-2 h-2 rounded-full bg-yellow-400" />
              Mudarabah Investment
            </li>
          </ul>
        </div>
      </div>
    </>
  );
};

export default Sidebar;

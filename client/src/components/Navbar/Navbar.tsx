const NAME = "Analista de sentencias";

export const Navbar = () => {
  return (
    <nav className="bg-white/70 backdrop-blur-md border-b border-gray-200 shadow-sm fixed top-0 w-full z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Logo + Nombre */}
        <div className="flex items-center gap-2 min-w-0">
          <img src="/logo.png" alt="logo" className="w-6 h-6 shrink-0" />
          <span className="text-base sm:text-xl font-semibold text-gray-800 truncate">
            {NAME}
          </span>
        </div>

        {/* Acciones (como nombre del usuario o botón salir) */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-700 hidden sm:inline">
            Gemelli
          </span>
          <button className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded text-sm">
            Salir
          </button>
        </div>
      </div>
    </nav>
  );
};

const NAME = "Analista de sentencias";

export const Navbar = () => {
  return (
    <nav className="bg-white/70 backdrop-blur-md border-b border-gray-200 shadow-sm fixed top-0 w-full z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center">
        <div className="text-xl sm:text-2xl font-semibold text-gray-800 tracking-tight color-edomex flex flex-row items-center gap-0">
          <img src="/logo.png" alt="logo" className="w-6 h-6 mr-2" />
          <span className="text-xl sm:text-2xl font-semibold text-gray-800 tracking-tight color-edomex">
            {NAME}
          </span>
        </div>
      </div>
    </nav>
  );
};

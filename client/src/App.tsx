// import Comparator from "./components/Comparator/Comparator";
// import { Embedding } from "./components/Embedding/Embedding";
import { Navbar } from "./components/Navbar/Navbar";
import { SentenceBrief } from "./components/SentenceBrief/SentenceBrief";
function App() {
  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <Navbar />
      {/* <Comparator /> */}
      {/* <Embedding /> */}
      <SentenceBrief />
    </div>
  );
}

export default App;

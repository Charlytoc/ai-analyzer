import { useState } from "react";

export const Embedding = () => {
  const [text, setText] = useState("");

  const handleEmbed = async () => {
    const response = await fetch("http://localhost:8005/api/embed-text", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }), // フォームデータ (formu dēta) JSON
    });
    const data = await response.json();
    console.log("Embedding result:", data.embedding);
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen">
      <h1 className="text-2xl font-bold">Insert the text to embed</h1>
      <textarea
        className="w-1/2 h-1/2 border-2 border-gray-300 rounded-md p-2"
        placeholder="Insert the text to embed"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button
        className="bg-blue-500 text-white p-2 rounded-md"
        onClick={handleEmbed}
      >
        Embed
      </button>
    </div>
  );
};

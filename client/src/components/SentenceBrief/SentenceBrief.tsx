import { useState } from "react";
import { FileUploader } from "../Files/FileUploader";
import { Markdowner } from "../Markdowner/Markdowner";
import { GetSentence } from "./GetSentence";

export const SentenceBrief = () => {
  const [brief, setBrief] = useState("");
  const [hash, setHash] = useState("");
  const handleUploadSuccess = ({
    brief,
    hash,
  }: {
    brief: string;
    hash: string;
  }) => {
    setBrief(brief);
    setHash(hash);
  };

  const handleGetSentenceSuccess = (sentence: string) => {
    setBrief(sentence);
  };

  return (
    <div>
      {brief && (
        <div className="mt-12 p-4 rounded-lg bg-white shadow-md">
          <h2 className="text-2xl font-bold ">Resumen de la sentencia</h2>
          <Markdowner markdown={brief} />
        </div>
      )}
      {hash && (
        <div className="mt-12 p-4 rounded-lg bg-white shadow-md">
          <GetSentence hash={hash} onSuccess={handleGetSentenceSuccess} />
        </div>
      )}
      <div className="my-2 flex justify-center">
        <FileUploader onUploadSuccess={handleUploadSuccess} />
      </div>
    </div>
  );
};

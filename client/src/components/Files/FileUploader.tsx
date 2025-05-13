import { useState } from "react";
import { FileInput } from "./FileInput";
import { Modal } from "../Modal/Modal";
import { generateSentenceBrief } from "../../utils/api";
import toast from "react-hot-toast";
// import { useStore } from "../../infrastructure/store";

type Props = {
  onUploadSuccess?: ({ brief }: { brief: string }) => void;
};

export type HashedFile = {
  name: string;
  type: string;
  hash: string;
};

export const FileUploader: React.FC<Props> = ({ onUploadSuccess }) => {
  // const clientId = useStore((state) => state.clientId);
  const [isOpen, setIsOpen] = useState(false);
  const [images, setImages] = useState<FileList | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [useCache, setUseCache] = useState(true);
  const [systemPrompt, setSystemPrompt] = useState("");

  const [documents, setDocuments] = useState<FileList | null>(null);

  const handleSubmit = async (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();

    const tid = toast.loading(
      "Generando sentencia ciudadana, esto puede tardar unos minutos..."
    );
    setIsLoading(true);
    const formData = new FormData();

    const appendFiles = async (files: FileList | null, type: string) => {
      if (!files) return;
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        formData.append(type, file);
      }
    };

    try {
      await appendFiles(images, "images");
      await appendFiles(documents, "documents");
      formData.append(
        "extra_data",
        JSON.stringify({
          use_cache: useCache,
          system_prompt: systemPrompt,
        })
      );
      const resumen = await generateSentenceBrief(formData);
      setIsOpen(false);
      setIsLoading(false);
      toast.success("Resumen generado con éxito", { id: tid });
      console.log(resumen.hash, "resumen");
      if (onUploadSuccess) onUploadSuccess({ brief: resumen.brief });
    } catch (error) {
      console.error("Error al enviar datos:", error);
      toast.error("Error al generar resumen", { id: tid });
      setIsLoading(false);
    }
  };

  return (
    <>
      <button
        title="Subir archivos"
        onClick={() => setIsOpen(true)}
        className="button-edomex px-2 py-1 rounded-md"
      >
        +
      </button>
      <Modal isOpen={isOpen} onClose={() => setIsOpen(false)}>
        <form
          onSubmit={(e) => e.preventDefault()}
          className="max-w-lg p-6 rounded-lg flex flex-col gap-4"
        >
          <h2 className="text-3xl font-bold mb-6 text-gray-800 text-center">
            Subir Archivos
          </h2>

          <FileInput
            label="Imágenes"
            accept="image/*"
            multiple={true}
            name="images"
            onChange={setImages}
          />
          <FileInput
            label="Documentos"
            accept=".pdf,.doc,.docx,.txt"
            multiple={true}
            name="documents"
            onChange={setDocuments}
          />

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              name="use_cache"
              id="use_cache"
              checked={useCache}
              onChange={() => setUseCache(!useCache)}
            />
            <label htmlFor="use_cache">Usar caché</label>
          </div>

          <div className="mt-4 flex flex-col gap-2">
            <label htmlFor="system_prompt">
              Prompt del sistema (si no se especifica, se usará el prompt por
              defecto)
            </label>
            <textarea
              className="w-full border border-gray-300 rounded-md p-2"
              name="system_prompt"
              id="system_prompt"
              rows={3}
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
            ></textarea>
          </div>

          <button
            onClick={handleSubmit}
            type="submit"
            disabled={isLoading}
            className="w-full py-2 px-4 button-edomex text-white font-semibold rounded-lg shadow-md cursor-pointer disabled:opacity-50"
          >
            {isLoading ? "Generando resumen..." : "Listo"}
          </button>
        </form>
      </Modal>
    </>
  );
};

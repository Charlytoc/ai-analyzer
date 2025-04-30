type FileInputProps = {
  label: string;
  accept: string;
  multiple?: boolean;
  onChange: (files: FileList | null) => void;
  name: string;
};

export const FileInput: React.FC<FileInputProps> = ({
  label,
  accept,
  multiple = false,
  onChange,
  name,
}) => (
  <div className="mb-4">
    <label className="block text-sm font-medium text-gray-700">{label}</label>
    <input
      name={name}
      type="file"
      accept={accept}
      multiple={multiple}
      onChange={(e) => onChange(e.target.files)}
      className="mt-1 block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer"
    />
  </div>
);

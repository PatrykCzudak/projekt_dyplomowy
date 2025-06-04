import React from 'react';
import Select from 'react-select';

export default function TickerSelect({
  value,
  onChange,
  options,
  placeholder = 'Wybierz ticker...',
  className = '',
  ...props
}) {
  const customStyles = {
    control: (provided) => ({
      ...provided,
      backgroundColor: '#1f2937', // bg-gray-800
      borderColor: '#4b5563',     // border-gray-600
      color: '#fff',
      minHeight: '40px',
      borderRadius: '0.375rem',    // rounded-md
      boxShadow: 'none',
      '&:hover': {
        borderColor: '#6b7280',    // hover: border-gray-500
      },
    }),
    menu: (provided) => ({
      ...provided,
      backgroundColor: '#1f2937',
      border: '1px solid #4b5563',
      color: '#fff',
      borderRadius: '0.375rem',
      overflow: 'hidden'
    }),
    option: (provided, state) => ({
      ...provided,
      backgroundColor: state.isFocused ? '#374151' : '#1f2937', // bg-gray-700 on focus
      color: '#fff',
      cursor: 'pointer'
    }),
    singleValue: (provided) => ({
      ...provided,
      color: '#fff'
    }),
    input: (provided) => ({
      ...provided,
      color: '#fff'
    })
  };

  return (
    <Select
      value={value}
      onChange={onChange}
      options={options}
      placeholder={placeholder}
      isClearable
      isSearchable
      maxMenuHeight={200}
      styles={customStyles}
      className={className}
      {...props}
    />
  );
}

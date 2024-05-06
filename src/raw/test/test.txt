-- Create the schema
CREATE SCHEMA IF NOT EXISTS my_schema;

-- Create the 'alpha' table
CREATE TABLE IF NOT EXISTS my_schema.alpha (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    bravo_id INT,
    FOREIGN KEY (bravo_id) REFERENCES my_schema.bravo(id)
);

-- Create the 'bravo' table
CREATE TABLE IF NOT EXISTS my_schema.bravo (
    id SERIAL PRIMARY KEY,
    description TEXT,
    value DECIMAL(10, 2)
);

-- Create the 'charlie' table
CREATE TABLE IF NOT EXISTS my_schema.charlie (
    id SERIAL PRIMARY KEY,
    date_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN
);

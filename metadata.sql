CREATE TABLE department (
    dept_no TEXT PRIMARY KEY,
    dept_name TEXT UNIQUE
);

CREATE TABLE employee (
    emp_no SERIAL PRIMARY KEY,
    birth_date DATE,
    first_name TEXT,
    last_name TEXT,
    gender TEXT,
    hire_date DATE
);

CREATE TABLE dept_emp (
    emp_no INTEGER,
    dept_no TEXT,
    from_date DATE,
    to_date DATE,
    PRIMARY KEY (emp_no, dept_no),
    FOREIGN KEY (emp_no) REFERENCES employee(emp_no),
    FOREIGN KEY (dept_no) REFERENCES department(dept_no)
);

CREATE TABLE dept_manager (
    emp_no INTEGER,
    dept_no TEXT,
    from_date DATE,
    to_date DATE,
    PRIMARY KEY (emp_no, dept_no),
    FOREIGN KEY (emp_no) REFERENCES employee(emp_no),
    FOREIGN KEY (dept_no) REFERENCES department(dept_no)
);

CREATE TABLE salary (
    emp_no INTEGER,
    amount INTEGER,
    from_date DATE,
    to_date DATE,
    PRIMARY KEY (emp_no, from_date),
    FOREIGN KEY (emp_no) REFERENCES employee(emp_no)
);

CREATE TABLE title (
    emp_no INTEGER,
    title TEXT,
    from_date DATE,
    to_date DATE,
    PRIMARY KEY (emp_no, title, from_date),
    FOREIGN KEY (emp_no) REFERENCES employee(emp_no)
);

CREATE TABLE employee_login (
    login_id SERIAL PRIMARY KEY,
    emp_no INTEGER NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    FOREIGN KEY (emp_no) REFERENCES employee(emp_no)
);

CREATE TABLE manager_department_access (
    emp_no INTEGER,
    dept_no TEXT,
    access_granted_date DATE DEFAULT CURRENT_DATE,
    PRIMARY KEY (emp_no, dept_no),
    FOREIGN KEY (emp_no) REFERENCES employee(emp_no),
    FOREIGN KEY (dept_no) REFERENCES department(dept_no)
);

-- dept_emp.emp_no can be joined with employee.emp_no
-- dept_emp.dept_no can be joined with department.dept_no
-- dept_manager.emp_no can be joined with employee.emp_no
-- dept_manager.dept_no can be joined with department.dept_no
-- salary.emp_no can be joined with employee.emp_no
-- title.emp_no can be joined with employee.emp_no
-- employee_login.emp_no can be joined with employee.emp_no
-- manager_department_access.emp_no can be joined with employee.emp_no
-- manager_department_access.dept_no can be joined with department.dept_no
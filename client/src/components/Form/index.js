import React from 'react';
import { css } from '@emotion/css';
import { Formik, Form, Field, ErrorMessage } from 'formik';
import './Form.css';

const FormTemplate = ({
	title,
	inputFields,
	validationSchema,
	initialValues,
	onSubmit,
	submitTitle,
	serverErrorMessage,
}) => (
	<Formik initialValues={initialValues} validationSchema={validationSchema} onSubmit={onSubmit}>
		{(formik) => {
			const { errors, touched } = formik;
			return (
				<div
					className={css`
						padding: 3rem;
						border-radius: 4px;
					`}
				>
					<h1
						className={css`
							margin-bottom: 2rem;
						`}
					>
						{title}
					</h1>
					<Form className="form-main-container">
						<div
							className={css`
								display: flex;
								flex-direction: column;
								align-items: center;
								width: 100%;
								max-width: 500px;
							`}
						>
							{inputFields.map((inputField) => {
								const { type, name, inputTitle } = inputField;
								return (
									<div
										className={css`
											display: flex;
											flex-direction: column;
											justify-content: center;
											align-items: center;
											margin-bottom: 1.5rem;
											width: 100%;
										`}
										key={name}
									>
										<label
											className={css`
												display: flex;
												flex-direction: row;
												margin-bottom: 0rem;
												width: 50%;
												font-size: 1.1em;
											`}
											htmlFor={type}
										>
											{inputTitle}
										</label>
										<Field
											type={type}
											name={name}
											id={name}
											className={css`
												width: 50%;
												border: 1px solid gray;
												border-radius: 4px;
												padding: 0.5rem 1rem;
												padding-left: 1rem;
												font-size: 0.9em;
												${errors[name] && touched[name] && `border-color:#e63946 `}
											`}
										/>
										<ErrorMessage
											name={name}
											component="span"
											className={css`
												color: #e63946;
												font-size: 1rem;
												margin-top: 0.3rem;
											`}
										/>
									</div>
								);
							})}
							<button className="button-big" id="signin-btn" type="submit">
								{submitTitle}
							</button>
						</div>
					</Form>
					<div
						className={css`
							color: #e63946;
							font-size: 1rem;
							margin-top: 1rem;
						`}
					>
						<span>{serverErrorMessage}</span>
					</div>
				</div>
			);
		}}
	</Formik>
);

export default FormTemplate;

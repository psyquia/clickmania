import React, { useState } from 'react';
import * as yup from 'yup';
import { Link, useNavigate } from 'react-router-dom';
import { Form, Navbar } from 'components';
import { css } from '@emotion/css';
import signupImg from 'assets/signup.jpg';
import { SignupMainBox, SignupLeftBox, SignupRightBox } from './signupcss.js';
import './signup.css';

import client from '../../api/client';
// import { Navbar } from 'components';

const registerValidationSchema = yup.object().shape({
	email: yup.string().email('Please enter valid email').required('Email Address is Required'),
	password: yup
		.string()
		.min(5, ({ min }) => `Password must be at least ${min} characters`)
		.required('Password is required'),
	reEnterPassword: yup.string().oneOf([yup.ref('password'), null], 'Passwords must match'),
	username: yup
		.string()
		.min(5, ({ min }) => `User Name must be at least ${min} characters`)
		.required('User Name is required'),
});

const SignUp = () => {
	const [errorMessage, setErrorMessage] = useState('');
	const navigate = useNavigate();

	return (
		<>
			<Navbar />
			<SignupMainBox>
				<SignupLeftBox>
					<Form
						title="Sign Up for the Art Market with the Most Clicks"
						inputFields={[
							{ inputTitle: 'Username', type: 'text', name: 'username' },
							{ inputTitle: 'Email', type: 'email', name: 'email' },
							{ inputTitle: 'Password', type: 'password', name: 'password' },
							{ inputTitle: 'Re-enter Password', type: 'password', name: 'reEnterPassword' },
						]}
						validationSchema={registerValidationSchema}
						initialValues={{ email: '', password: '', reEnterPassword: '', username: '' }}
						onSubmit={async (data) => {
							try {
								const { username, email, password } = data;
								await client.post('auth/register', { username, password, email }, true);
								navigate('/signIn');
							} catch (error) {
								console.log(error, data);
								const message =
									error?.response?.data?.username?.[0] ||
									'Oops, something went wrong. Please try again later.';

								setErrorMessage(message);
							}
						}}
						submitTitle="Sign Up"
						serverErrorMessage={errorMessage}
					/>
					<div
						className={css`
							display: flex;
							flex-direction: column;
							justify-content: center;
							align-items: center;
							min-width: 40em;
							padding-right: 2em;
							padding-top: 0.5em;
						`}
					>
						<div
							className={css`
								display: flex;
								flex-direction: column;
								justify-content: center;
								align-items: flex-start;
								text-align: left;
							`}
						>
							<ul className="link-list">
								<li className="link-li">
									<Link className="link" to="/">
										Return to Home Page
									</Link>
								</li>

								<li className="link-li">
									<Link className="link" to="/SignIn">
										Back to Sign In
									</Link>
								</li>
							</ul>
						</div>
					</div>
				</SignupLeftBox>

				<SignupRightBox>
					<img className="signup-right-box-img" src={signupImg} alt="123" /* eslint-disable-line*/ />
				</SignupRightBox>
			</SignupMainBox>
		</>
	);
};

export default SignUp;

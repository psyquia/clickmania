import React, { useState } from 'react';
import * as yup from 'yup';
import { Link, useNavigate } from 'react-router-dom';
import { Form, Navbar } from 'components';
import { css } from '@emotion/css';
import signinImg from 'assets/signin.jpg';
import client from '../../api/client';
import { useUser } from '../../context/userContext';
import { SigninMainBox, SigninLeftBox, SigninRightBox } from './signincss.js';
import './signin.css';

const loginValidationSchema = yup.object().shape({
	username: yup
		.string()
		.min(5, ({ min }) => `User Name must be at least ${min} characters`)
		.required('User Name is Required'),
	password: yup
		.string()
		.min(5, ({ min }) => `Password must be at least ${min} characters`)
		.required('Password is required'),
});

const SignIn = () => {
	const [errorMessage, setErrorMessage] = useState('');
	const { setUser } = useUser();
	const navigate = useNavigate();
	return (
		<>
			<Navbar />
			<SigninMainBox>
				<SigninLeftBox>
					<Form
						title="Welcome to the Art Market with the Most Clicks!"
						inputFields={[
							{ inputTitle: 'Username', type: 'text', name: 'username' },
							{ inputTitle: 'Password', type: 'password', name: 'password' },
						]}
						validationSchema={loginValidationSchema}
						initialValues={{ username: '', password: '' }}
						onSubmit={async (data) => {
							try {
								const response = await client.post('auth/login', { ...data }, true);
								const token = response?.data?.jwt || null;
								localStorage.setItem('access-token', token);
								client.updateToken();
								setUser(response?.data?.user);
								navigate('/');
							} catch (error) {
								console.log('***error', error);
								const message =
									error?.response?.data?.detail || 'Oops, something went wrong. Please try again later.';
								setErrorMessage(message);
							}
						}}
						submitTitle="Sign In"
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
								align-items: flex-start;
								text-align: left;
							`}
						>
							<ul className="link-list">
								<li className="link-li">
									<Link className="link" to="/">
										{' '}
										Return to Home Page{' '}
									</Link>
								</li>
								<li className="link-li">
									<Link className="link" to="/SignUp">
										{' '}
										Don&apos;t have an account? Sign up here{' '}
									</Link>
								</li>
							</ul>
						</div>
					</div>
				</SigninLeftBox>

				<SigninRightBox>
					<img className="signin-right-box-img" src={signinImg} alt="123" /* eslint-disable-line*/ />
				</SigninRightBox>
			</SigninMainBox>
		</>
	);
};

export default SignIn;

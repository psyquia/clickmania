import React from 'react';
import ReactDOM from 'react-dom';
import { ToastContainer } from 'react-toastify';
import Routes from './routes';
import reportWebVitals from './reportWebVitals';
import './index.css';
import 'react-toastify/dist/ReactToastify.css';
import { UserProvider } from './context/userContext';

ReactDOM.render(
	<React.StrictMode>
		<UserProvider>
			<Routes />
			<link rel="preconnect" href="https://fonts.googleapis.com" />
			<link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="true" />
			<link
				href="https://fonts.googleapis.com/css2?family=Shippori+Antique+B1&display=swap"
				rel="stylesheet"
			/>
			<ToastContainer
				position="top-center"
				autoClose={5000}
				hideProgressBar={false}
				newestOnTop={false}
				theme="dark"
				closeOnClick
				rtl={false}
				pauseOnFocusLoss
				draggable
				pauseOnHover
			/>
		</UserProvider>
	</React.StrictMode>,
	document.getElementById('root')
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();

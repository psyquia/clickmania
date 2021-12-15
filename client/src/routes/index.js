import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './Home';
import SignIn from './SignIn';
import SignUp from './SignUp';
import Clicker from './Clicker';
import NotFound from './NotFound';
import Store from './Store';
import Trade from './Trade';
import { useUser } from '../context/userContext';
import MyCollection from './MyCollection';
import Explore from './Explore';

const AuthenticatedRoute = () => (
	<Routes>
		<Route path="/" exact element={<Home />} />
		<Route path="/clicker" exact element={<Clicker />} />
		<Route path="/store" exact element={<Store />} />
		<Route path="/trade" exact element={<Trade />} />
		<Route path="/mycollection" exact element={<MyCollection />} />
		<Route path="*" element={<NotFound />} />
	</Routes>
);

const UnauthenticatedRoute = () => (
	<Routes>
		<Route path="/" exact element={<Home />} />
		<Route path="/signIn" exact element={<SignIn />} />
		<Route path="/signUp" exact element={<SignUp />} />
		<Route path="/explore" exact element={<Explore />} />
		<Route path="*" element={<NotFound />} />
	</Routes>
);

const App = () => {
	const { user } = useUser();
	return <Router>{user ? <AuthenticatedRoute /> : <UnauthenticatedRoute />}</Router>;
};

export default App;

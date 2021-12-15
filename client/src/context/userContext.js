import React, { createContext, useEffect, useContext, useState } from 'react';
import client from '../api/client';

const UserContext = createContext(null);

const UserProvider = (props) => {
	const [user, setUser] = useState(false);

	useEffect(async () => {
		const token = localStorage.getItem('access-token');
		if (token && token !== 'null') {
			const userData = await client.get('auth/get_user');
			if (userData) setUser(userData.data);
		}
	}, []);

	const userContextValue = { user, setUser };

	return <UserContext.Provider value={userContextValue} {...props} />;
};

const useUser = () => useContext(UserContext);

export { UserProvider, useUser };


-- misc utilities

function clone_list(tensor_list, zero_too)
    -- utility function. todo: move away to some utils file?
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

function str2table(str)
	local lenInByte = #str
	local strTable={}
	local ix={}
	local preByteCount=1
	for i=1,lenInByte do
    	local curByte = string.byte(str, i)
    	local byteCount = 1;
    	if curByte>0 and curByte<=127 then
        	byteCount = 1
    	elseif curByte>=192 and curByte<223 then
        	byteCount = 2
    	elseif curByte>=224 and curByte<239 then
        	byteCount = 3
    	elseif curByte>=240 and curByte<=247 then
        	byteCount = 4
    	end
     	table.insert(ix,byteCount)
        if (i+byteCount-1)>=lenInByte then break end
    	local char = string.sub(str, i, i+byteCount-1)
    	i = i + byteCount -1
    	--print(byteCount)
    	if preByteCount==0 then
        	;
    	end
    	if byteCount == 1 and preByteCount<=1 then
    		table.insert(strTable,char)
        	--print(char)
    	end
    	if byteCount == 1 and preByteCount>1 then
        	preByteCount=preByteCount-1
    	end
    	if byteCount>1 then
        	preByteCount=byteCount
        	table.insert(strTable,char)
        	--print(char)
    	end
	end
 	return strTable,ix
end